# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

from nemo_skills.code_execution import extract_code_to_execute, format_code_output
from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.inference.model.utils import trim_after_stop_phrases
from nemo_skills.utils import get_logger_name, nested_dataclass

from .base import BaseModel
from .utils import trim_after_stop_phrases

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class CodeExecutionConfig:
    max_code_output_characters: int = 1000
    code_execution_timeout: float = 10.0
    code_execution_language: str = 'ipython'
    code_execution_headers: list[str] = field(default_factory=lambda: [])
    max_code_executions: int = 8
    sandbox_traceback_verbosity: str = 'plain'  # could be plain, context, verbose, or minimal
    add_remaining_code_executions: bool = False

    # NOTE: prompt template is required for properly format harmony prompt in text completion
    user_begin: str = "<|start|>user<|message|>"
    user_end: str = "<|end|>"
    assistant_begin: str = "<|start|>assistant"
    assistant_end: str = "<|end|>"
    final_answer_begin: str = "<|start|>assistant<|channel|>final<|message|>"
    final_answer_end: str = "<|return|>"

class CodeExecutionWrapper:
    def __init__(self, model: BaseModel, sandbox: Sandbox, config: CodeExecutionConfig):
        self.model = model
        self.sandbox = sandbox
        self.config = config

    async def _generate_single(
        self,
        prompt: str | list[dict],
        code_begin: str,
        code_end: str,
        code_output_begin: str,
        code_output_end: str,
        code_output_format: str,
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: float | int | None = 14400,  # None is 10min
        max_code_executions: int | None = None,  # if not None, will override self.config.max_code_executions
        stream: bool = False,
        extra_body: dict = None,
    ):
        # Handle OpenAI-style dictionary prompts
        is_openai_format = not isinstance(prompt, str)
        
        if top_logprobs is not None:  # TODO: add this
            raise NotImplementedError("top_logprobs is not supported yet.")

        if stream:
            return await self._stream_single(
                prompt=prompt,
                code_begin=code_begin,
                code_end=code_end,
                code_output_begin=code_output_begin,
                code_output_end=code_output_end,
                code_output_format=code_output_format,
                tokens_to_generate=tokens_to_generate,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                stop_phrases=stop_phrases,
                timeout=timeout,
                max_code_executions=max_code_executions,
                extra_body=extra_body,
            )
        
        # this is the max number of code executions
        effective_max_code_executions = self.config.max_code_executions
        if max_code_executions is not None:
            effective_max_code_executions = max_code_executions

        # making a copy of prompts to not corrupt original data
        if is_openai_format:
            new_prompt = copy.deepcopy(prompt)
        else:
            new_prompt = copy.deepcopy(prompt)

        # set up timeout limit for the generation loop. 
        # this includes all the time spent calling the LLM, executing the code, etc. 
        start_time = int(time.time())

        stop_phrases = stop_phrases or []

        request = {
            "prompt": new_prompt,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_phrases": stop_phrases + [code_end],
            "timeout": timeout,
            "extra_body": extra_body,
        }
        session_id = None
        code_rounds_executed = 0
        total_num_generated_tokens = 0
        generation_time = 0
        code_execution_time = 0
        stopped_on_repetition = False

        #NOTE: constrain generation of effective max executions plus some buffer. 
        # ideally, each generation produces a block, then max_generation_rounds = effective_max_code_executions
        # however, some models like to pause and think, which means that some generations will not contain a code block. 
        # Hence we set a another limit, which is max_generation_rounds. This is the max number of generations we can have, for this input problem. 
        # The generation stops when it reaches effective_max_code_executsion, OR max_generation_rounds, whichever comes first. 
        generation_index = 0
        max_generation_rounds= effective_max_code_executions*2 # this is arbitrary. 

        while code_rounds_executed <= effective_max_code_executions and generation_index <= max_generation_rounds:

            # if we timeout for this problem, we break out of the loop
            generation_time_start = time.time()
            if timeout is not None:
                # updating timeout to account for the time already spent
                new_timeout = int(timeout - (time.time() - start_time))
                request["timeout"] = new_timeout
                if request['timeout'] <= 0:
                    break

            # here we send the prompt to LLM
            output_dict = await self.model.generate_async(**request, remove_stop_phrases=False)
            print(f"--------------DEBUGGING generation_index: {generation_index}: output_dict-------------")
            print(output_dict)

            output, num_generated_tokens = output_dict['generation'], output_dict.get('num_generated_tokens', 0)
            # no need to do anything with this as the code below should just exit, so that's only for logging
            stopped_on_repetition = output_dict.get('stopped_on_repetition', False)

            # NOTE: If we see there is more code_begin than code_end, the code generation was stopped, and the last token was `code_end`
            # we add it back for code parsing. 
            if output_dict.get('finish_reason') == 'stop':
                if output.count(code_end) + 1 == output.count(code_begin):
                    output += code_end

            print(f"--------------DEBUGGING generation_index: {generation_index}: output-------------")
            print(output)

            # Update the prompt with the LLM's response (e.g. code generation or thinking)
            if is_openai_format:
                # openai format is a list
                request['prompt'].append({'role': 'assistant', 'content': output})
            else:
                # non-openai format (text completion) is formatted string. 
                request['prompt'] += output

            # if it's the extra iteration, we don't execute the code block and just finish
            if generation_index == effective_max_code_executions:
                print(f"--------------DEBUGGING generation_index: {generation_index}: reached effective_max_code_executions, break-------------")
                break
            
            # adjusting requested tokens to account for what has been generated already
            request['tokens_to_generate'] -= num_generated_tokens
            total_num_generated_tokens += num_generated_tokens
            generation_time += int(time.time() - generation_time_start)
            # TODO: currently we don't account for tokens in the code output that we add to the prompt
            if request['tokens_to_generate'] <= 0:
                print(f"--------------DEBUGGING generation_index: {generation_index}: reached max tokens, break-------------")
                break

            # .rfind(code_end, 0, -1) searches for the second-to-last occurrence of code_end and checks
            # that the last code_begin is not closed to ensure that we are inside the code block
            if output.endswith(code_end) and output.rfind(code_begin) > output.rfind(code_end, 0, -1):
                print(f"--------------DEBUGGING generation_index: {generation_index}: Found code------------")
                
                # extract and execute the code block
                code_execution_time_start, execution_dict, session_id = await self.execute_generated_code(
                    prompt, code_begin, code_end, output, session_id
                )
                print(f"--------------DEBUGGING generation_index: {generation_index}: code execution_dict-------------")
                print(execution_dict)
                
                # calculate the remaining code executions, and tell the model how many are left. 
                remaining_code_executions = None
                if self.config.add_remaining_code_executions:
                    remaining_code_executions = effective_max_code_executions - generation_index - 1
                
                # adding code output to the prompt
                code_output = format_code_output(
                    execution_dict, code_output_begin, code_output_end, code_output_format, remaining_code_executions
                )
                print(f"--------------DEBUGGING generation_index: {generation_index}: code_output-------------")
                print(code_output)

                # update the prompt with the code output
                if is_openai_format:
                    # -1 for the assistant message, which is the last message. 
                    request['prompt'][-1]['content'] += code_output
                else:
                    request['prompt'] += code_output

                # calculate code execution time. 
                code_execution_time += int(time.time() - code_execution_time_start)
                code_rounds_executed += 1

            else: 
                # NOTE: if no code was generated, we just ignore. Sometimes model is just thinking. 
                pass
            
            # NOTE: regardless of whether code was generated, we increase generation index, to track the number of generations. 
            generation_index += 1

            # NOTE: we need to check if the final answer has been generated. 
            # If so, we should break out the loop to prevent furthur interactions with the LLM
            if is_openai_format:
                # the default final answer begin is "Final Answer:"
                if self.config.final_answer_begin in request['prompt'][-1]['content']:
                    print(f"--------------DEBUGGING generation_index: {generation_index}: Final answer found, break-------------")
                    break
            else:
                if self.config.final_answer_begin in output:
                    print(f"--------------DEBUGGING generation_index: {generation_index}: Final answer found, break-------------")
                    break

            # NOTE: model hasn't generated the final answer yet, so we should continue with generation. 
            # some models (like gpt-oss) likes a confirmation signal from the user so that LLM can continue generating
            if is_openai_format:
                request['prompt'].append({'role': 'user', 'content': "continue"})
            else:
                continue_signal = f"{self.config.user_begin}continue{self.config.user_end}"
                request['prompt'] += continue_signal
        
        # clean up after generaiton is done
        # removing original prompt and returning the generation
        if is_openai_format:
            generation = "\n".join(msg['content'] for msg in request['prompt'] if msg['role'] == 'assistant')
        else:
            # NOTE: cut off the prompt, and remove the final continue signal string from the user
            # generation = request['prompt'][len(prompt):].replace(continue_signal, "")
            # for deugging, retain the use interactions in the generation: 
            assistant_begin = "<|start|>assistant"
            generation = request['prompt'][len(prompt)-len(assistant_begin):]

        return {
            'generation': generation,
            'code_rounds_executed': code_rounds_executed,
            'num_generated_tokens': total_num_generated_tokens,
            'generation_time': generation_time,
            'code_execution_time': code_execution_time,
            'stopped_on_repetition': stopped_on_repetition,
        }

    async def execute_generated_code(self, input_prompt, code_begin, code_end, output, session_id):
        code_execution_time_start = time.time()
        header = '\n'.join(self.config.code_execution_headers)
        code_block = extract_code_to_execute(output, code_begin, code_end)
        extracted_code = f'{header}{code_block}'
        execution_dict, session_id = await self.sandbox.execute_code(
            generated_code=extracted_code,
            language=self.config.code_execution_language,
            timeout=self.config.code_execution_timeout,
            max_output_characters=self.config.max_code_output_characters,
            session_id=session_id,
            traceback_verbosity=self.config.sandbox_traceback_verbosity,
        )

        return code_execution_time_start, execution_dict, session_id

    async def generate_async(
        self,
        prompt: str | list[dict],
        code_begin: str,
        code_end: str,
        code_output_begin: str,
        code_output_end: str,
        code_output_format: str,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
        top_logprobs: int | None = None,
        timeout: float | int | None = 14400,  # None is 10min
        max_code_executions: int | None = None,
        stream: bool = False,
        extra_body: dict = None,
    ) -> list[dict]:
        """For any generation parameter you can specify a list of values that needs to match the number of prompts.

        Not every server supports that, so make sure to override this method directly if that's not the case.
        """
        if top_logprobs is not None:  # TODO: add this
            raise NotImplementedError("top_logprobs is not supported yet.")

        kwargs = {
            'code_begin': code_begin,
            'code_end': code_end,
            'code_output_begin': code_output_begin,
            'code_output_end': code_output_end,
            'code_output_format': code_output_format,
            'tokens_to_generate': tokens_to_generate,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'min_p': min_p,
            'repetition_penalty': repetition_penalty,
            'random_seed': random_seed,
            'stop_phrases': stop_phrases,
            "timeout": timeout,
            "max_code_executions": max_code_executions,
            "stream": stream,
            "extra_body": extra_body,
        }

        request = {key: value for key, value in kwargs.items()}
        request['prompt'] = prompt

        output = await self._generate_single(**request)
        self.model._maybe_apply_stop_phrase_removal(output, remove_stop_phrases, stop_phrases)

        return output

    async def _stream_single(
        self,
        prompt: str | list[dict],
        code_begin: str,
        code_end: str,
        code_output_begin: str,
        code_output_end: str,
        code_output_format: str,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        timeout: float | int | None = 14400,  # None is 10min,
        max_code_executions: int | None = None,
        extra_body: dict = None,
    ):
        """
        Helper method, that implements streaming generation.
        """
        # Handle OpenAI-style dictionary prompts
        is_openai_format = not isinstance(prompt, str)

        effective_max_code_executions = self.config.max_code_executions
        if max_code_executions is not None:
            effective_max_code_executions = max_code_executions

        stop_phrases = stop_phrases or []

        request = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'min_p': min_p,
            'repetition_penalty': repetition_penalty,
            'random_seed': random_seed,
            'stop_phrases': stop_phrases + [code_end],
            'timeout': timeout,
            'tokens_to_generate': tokens_to_generate,
            'stream': True,
            'extra_body': extra_body,
        }

        current_full_prompt = copy.deepcopy(prompt)
        session_id = None  # For sandbox state continuity
        for generation_index in range(effective_max_code_executions + 1):

            # NOTE:
            print("--------------DEBUGGING: running the generation loop-------------")
            print("generation_index: ", generation_index)
            print("effective_max_code_executions: ", effective_max_code_executions)

            model_token_iterator = self.model._generate_single(prompt=current_full_prompt, **request)

            current_output_segment = ""
            num_generated_tokens = 0
            for chunk in model_token_iterator:
                yield chunk
                current_output_segment += chunk['generation']
                num_generated_tokens += 1

            request['tokens_to_generate'] -= num_generated_tokens
            if request['tokens_to_generate'] <= 0:
                break
            if not current_output_segment:
                break

            # openai don't show what stop word was triggered, so we assume that it was `code_end`
            # if there's an unfinished code block
            if is_openai_format and chunk.get('finish_reason') == 'stop':
                if current_output_segment.count(code_end) + 1 == current_output_segment.count(code_begin):
                    current_output_segment += code_end
            # Update the prompt based on format
            if is_openai_format:
                current_full_prompt.append({'role': 'assistant', 'content': current_output_segment})
                current_full_prompt.append({'role': 'user', 'content': "continue"})
            else:
                current_full_prompt += current_output_segment

            if generation_index == effective_max_code_executions:
                # This was the last iteration, intended for final text generation after all code executions.
                break

            # NOTE: 
            print("--------------DEBUGGING: current_full_prompt-------------")
            print(current_full_prompt)

            if current_output_segment.endswith(code_end) and current_output_segment.rfind(
                code_begin
            ) > current_output_segment.rfind(code_end, 0, -1):
                # NOTE: 
                print("--------------DEBUGGING: executing code-------------")
                execution_dict, session_id = await self.sandbox.execute_code(
                    generated_code=extract_code_to_execute(current_output_segment, code_begin, code_end),
                    language=self.config.code_execution_language,
                    timeout=self.config.code_execution_timeout,
                    max_output_characters=self.config.max_code_output_characters,
                    session_id=session_id,
                    traceback_verbosity=self.config.sandbox_traceback_verbosity,
                )
                # NOte:
                print("--------------DEBUGGING: execution_dict-------------")
                print(execution_dict)

                remaining_code_executions = None
                if self.config.add_remaining_code_executions:
                    remaining_code_executions = effective_max_code_executions - generation_index - 1

                formatted_code_output = format_code_output(
                    execution_dict, code_output_begin, code_output_end, code_output_format, remaining_code_executions
                )
                # NOTE: 
                print("--------------DEBUGGING: formatted_code_output--------------")
                print(formatted_code_output)

                yield {'generation': formatted_code_output}  # Yield the entire formatted code output as one chunk

                # Append executed code's output to the prompt
                if is_openai_format:
                    current_full_prompt[-2]['content'] += formatted_code_output
                else:
                    current_full_prompt += formatted_code_output

            # NOTE: I didn't not implement the parllel funciton to limit the max_consecutive_no_code_generations
            # as in _generate_single, because it's not needed for streaming generation
            else:
                break
