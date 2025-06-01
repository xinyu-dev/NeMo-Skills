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


import abc
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

import httpx
import openai
import requests
from openai import DefaultHttpxClient, Stream

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def trim_after_stop_phrases(text: str, stop_phrases: list[str]) -> str:
    """Removes everything after the last stop token."""
    if not stop_phrases:
        return text
    # Escape all special characters in stop phrases
    escaped_stop_phrases = [re.escape(sp) for sp in stop_phrases]
    return re.split("|".join(escaped_stop_phrases), text, maxsplit=1)[0]


class BaseModel(abc.ABC):
    """Base model class for handling requests to the inference server.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the inference server.
        port: Optional[str] = '5000' - Port of the inference server.
            Only required if handle_code_execution is True.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access
            Can also be specified through NEMO_SKILLS_SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through NEMO_SKILLS_SSH_KEY_PATH env var.
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: str = '5000',
        ssh_server: str | None = None,
        ssh_key_path: str | None = None,
    ):
        self.server_host = host
        self.server_port = port
        self.ssh_server = ssh_server
        self.ssh_key_path = ssh_key_path
        if ssh_server is None:
            self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER")
        if ssh_key_path is None:
            self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH")

        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            self.requests_lib = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
        else:
            # TODO: switch to httpx
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_maxsize=1500, pool_connections=1500, max_retries=3)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            self.requests_lib = session

        self.gen_id_to_params = {}
        self.gen_id_to_future = {}

        self.executor = ThreadPoolExecutor(max_workers=1024)  # is this too much?

    @abc.abstractmethod
    def _generate_single(
        self,
        prompt: str | dict,
        tokens_to_generate: int | list[int],
        temperature: float | list[float],
        top_p: float | list[float],
        top_k: int | list[int],
        min_p: float | list[float],
        repetition_penalty: float | list[float],
        random_seed: int | list[int],
        stop_phrases: list[str] | list[list[str]] | None,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stream: bool = False,
    ) -> dict:
        """If the engine supports inflight-batching of requests, you only need to define this method.

        We will call it in threads on the list of prompts.
        """
        pass

    def preprocess_request(self, request: dict):
        """Just a small utility to pre-process some of the parameters of request."""
        # temperature of 0 means greedy, but it's not always supported by the server
        # so setting explicit greedy parameters instead
        if request["temperature"] == 0:
            request["temperature"] = 1.0
            request["top_k"] = 1
            request["top_p"] = 1.0

    def generate_async(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int | list[int] = 2048,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        top_logprobs: int | list[int] | None = None,
        timeout: int | list[int] | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
    ) -> list[dict]:
        """Returns a list of generation ids that can be later queried with get_generation calls."""
        kwargs = {
            'tokens_to_generate': tokens_to_generate,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'min_p': min_p,
            'repetition_penalty': repetition_penalty,
            'random_seed': random_seed,
            'stop_phrases': stop_phrases,
            'top_logprobs': top_logprobs,
            'timeout': timeout,
            'stream': stream,
        }
        for key, value in kwargs.items():
            is_list = False
            if key == 'stop_phrases' and (value and isinstance(value[0], list)):
                is_list = True
            if key != 'stop_phrases' and isinstance(value, list):
                is_list = True
            if is_list and len(value) != len(prompts):
                raise ValueError(f"Length of {key} should match the number of prompts.")
            if not is_list:
                kwargs[key] = [value for _ in range(len(prompts))]

        gen_ids = []
        for request_idx in range(len(prompts)):
            # Prepare request
            request = {key: value[request_idx] for key, value in kwargs.items()}
            request['prompt'] = prompts[request_idx]
            self.preprocess_request(request)

            # Generate a unique generation ID
            gen_id = str(uuid.uuid4())
            gen_ids.append(gen_id)

            # Update global dictionaries tracking the progress of generations
            self.gen_id_to_future[gen_id] = self.executor.submit(self._generate_single, **request)
            self.gen_id_to_params[gen_id] = (kwargs["stop_phrases"][request_idx], remove_stop_phrases)

        return gen_ids

    def get_generations(self, generation_ids: list[str]) -> list[dict]:
        generations = []
        for generation_id in generation_ids:
            if generation_id not in self.gen_id_to_future:
                raise ValueError(f"Generation id {generation_id} not found.")

            stop_phrases, remove_stop_phrases = self.gen_id_to_params[generation_id]
            future = self.gen_id_to_future[generation_id]
            if not future.done():
                output = {'generation': None}
            else:
                output = future.result()
                del self.gen_id_to_future[generation_id]
                del self.gen_id_to_params[generation_id]

            if remove_stop_phrases:
                if isinstance(output, dict) and output['generation'] is not None:
                    output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

            generations.append(output)

        return generations

    def generate(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int | list[int] = 2048,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        top_logprobs: int | list[int] | None = None,
        timeout: int | list[int] | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
    ) -> list[dict]:
        """For any generation parameter you can specify a list of values that needs to match the number of prompts.

        Not every server supports that, so make sure to override this method directly if that's not the case.
        """
        generation_ids = self.generate_async(
            prompts=prompts,
            tokens_to_generate=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_phrases=stop_phrases,
            top_logprobs=top_logprobs,
            remove_stop_phrases=remove_stop_phrases,
            stream=stream,
        )
        all_generations = [None] * len(prompts)
        while True:
            remaining_ids = [generation_id for generation_id in generation_ids if generation_id is not None]
            if len(remaining_ids) == 0:
                break
            remaining_positions = [
                idx for idx, generation_id in enumerate(generation_ids) if generation_id is not None
            ]
            generations = self.get_generations(remaining_ids)
            for gen_pos, gen_dict in zip(remaining_positions, generations):
                if isinstance(gen_dict, Generator) or gen_dict['generation'] is not None:  # will be None until done
                    generation_ids[gen_pos] = None
                    all_generations[gen_pos] = gen_dict
                    # trtllm always return these fields so we need to remove them if not requested
                    if isinstance(gen_dict, dict) and top_logprobs is None:
                        gen_dict.pop('tokens', None)
                        gen_dict.pop('logprobs', None)

            time.sleep(1)

        return all_generations


class TRTLLMModel(BaseModel):
    def _generate_single_base(
        self,
        prompt: str | dict,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stop_phrases: list[str] | None = None,
        generate_endpoint: str = "generate",
        stream: bool = False,
    ) -> list[dict]:
        if isinstance(prompt, dict):
            raise NotImplementedError("trtllm server does not support OpenAI \"messages\" as prompt.")
        if top_logprobs is not None and top_logprobs > 1:
            raise NotImplementedError("This code does not support `top_logprobs` > 1.")
        if stream:
            raise NotImplementedError("trtllm server does not support streaming.")
        if generate_endpoint not in ["generate", "generate_async"]:
            raise ValueError(f"Invalid generate endpoint: {generate_endpoint}")

        if stop_phrases is None:
            stop_phrases = []

        request = {
            "prompt": prompt,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "top_p_min": min_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_words_list": stop_phrases,
            "top_logprobs": top_logprobs,
            "timeout": timeout,
        }
        output_dict = self.requests_lib.put(
            url="http://{}:{}/{}".format(self.server_host, self.server_port, generate_endpoint),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()

        if generate_endpoint == "generate":
            return output_dict
        else:
            return output_dict['generation_id']

    def _generate_single_async(self, prompt: str | dict, **kwargs):
        """Asynchronous generation."""
        return self._generate_single_base(prompt, generate_endpoint="generate_async", **kwargs)

    def _generate_single(self, prompt: str | dict, **kwargs):
        """Synchronous generation."""
        return self._generate_single_base(prompt, generate_endpoint="generate", **kwargs)

    def generate_async(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int | list[int] = 2048,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        top_logprobs: int | list[int] | None = None,
        timeout: int | list[int] | None = None,
        stop_phrases: list[str] | list[list[str]] | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
    ) -> list[dict]:
        """For any generation parameter you can specify a list of values that needs to match the number of prompts.

        Not every server supports that, so make sure to override this method directly if that's not the case.
        """
        kwargs = {
            'tokens_to_generate': tokens_to_generate,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'min_p': min_p,
            'repetition_penalty': repetition_penalty,
            'random_seed': random_seed,
            'stop_phrases': stop_phrases,
            'top_logprobs': top_logprobs,
            'timeout': timeout,
            'stream': stream,
        }
        for key, value in kwargs.items():
            is_list = False
            if key == 'stop_phrases' and (value and isinstance(value[0], list)):
                is_list = True
            if key != 'stop_phrases' and isinstance(value, list):
                is_list = True
            if is_list and len(value) != len(prompts):
                raise ValueError(f"Length of {key} should match the number of prompts.")
            if not is_list:
                kwargs[key] = [value for _ in range(len(prompts))]

        # futures are not really necessary here unless the number of prompts is huge
        # as we get http reply right away with generation id from the trtllm server
        futures = []
        for request_idx in range(len(prompts)):
            request = {key: value[request_idx] for key, value in kwargs.items()}
            request['prompt'] = prompts[request_idx]
            self.preprocess_request(request)
            futures.append(self.executor.submit(self._generate_single_async, **request))
        outputs = [future.result() for future in futures]

        new_gen_id_to_params = {
            gen_id: (req_stop_phrases, remove_stop_phrases)
            for gen_id, req_stop_phrases in zip(outputs, kwargs["stop_phrases"])
        }

        self.gen_id_to_params.update(new_gen_id_to_params)

        return outputs

    def cancel_generations(self, generation_ids: list[str]) -> list[str]:
        statuses = []
        for generation_id in generation_ids:
            request = {
                "generation_id": generation_id,
            }
            output_dict = self.requests_lib.put(
                url="http://{}:{}/cancel_generation".format(self.server_host, self.server_port),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
            statuses.append(output_dict["status"])

        return statuses

    def get_generations(self, generation_ids: list[str]) -> list[dict]:
        generations = []
        for generation_id in generation_ids:
            request = {
                "generation_id": generation_id,
            }
            output = self.requests_lib.put(
                url="http://{}:{}/get_generation".format(self.server_host, self.server_port),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
            stop_phrases, remove_stop_phrases = self.gen_id_to_params[generation_id]
            if remove_stop_phrases:
                if output['generation'] is not None:
                    output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

            generations.append(output)

        return generations


class NemoModel(BaseModel):
    def _generate_single(
        self,
        prompt: str | dict,
        tokens_to_generate: int | list[int] = 512,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        min_p: float = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stop_phrases: list[str] | list[list[str]] | None = None,
        stream: bool = False,
    ) -> list[dict]:
        """If the engine supports inflight-batching of requests, you only need to define this method.

        We will call it in threads on the list of prompts.
        """
        if min_p > 0:
            raise NotImplementedError("Nemo server does not support min_p parameter.")
        if top_logprobs is not None:
            raise NotImplementedError("Nemo server does not support top_logprobs parameter.")
        if timeout is not None:
            raise NotImplementedError("Nemo server does not support timeout parameter.")
        if isinstance(prompt, dict):
            raise NotImplementedError("NeMo server does not support OpenAI \"messages\" as prompt.")
        if stream:
            raise NotImplementedError("NeMo server does not support streaming.")
        if stop_phrases is None:
            stop_phrases = []
        request = {
            "sentences": [prompt],
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "end_strings": ["<|endoftext|>"] + stop_phrases,
        }
        generations = self.requests_lib.put(
            url="http://{}:{}/generate".format(self.server_host, self.server_port),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()
        # we need to remove the original prompt as nemo always returns it
        output = generations['sentences'][0]
        # when the prompt starts from special tokens like bos, nemo will remove them,
        # so we need this hack to find where to start the cut
        begin_idx = 0
        while begin_idx < len(prompt) and not prompt[begin_idx:].startswith(output[:20]):
            begin_idx += 1
        output = {'generation': output[(len(prompt) - begin_idx) :]}
        return output

    def generate(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
    ) -> list[dict]:
        # we are overriding generate directly, since nemo doesn't support inflight batching
        if min_p > 0:
            raise NotImplementedError("Nemo server does not support min_p parameter.")
        if top_logprobs is not None:
            raise NotImplementedError("Nemo server does not support top_logprobs parameter.")
        if timeout is not None:
            raise NotImplementedError("Nemo server does not support timeout parameter.")

        if isinstance(prompts[0], dict):
            raise NotImplementedError("NeMo server does not support OpenAI \"messages\" as prompt.")
        if stop_phrases is None:
            stop_phrases = []
        request = {
            "sentences": prompts,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "end_strings": ["<|endoftext|>"] + stop_phrases,
            "stream": stream,
        }
        self.preprocess_request(request)
        generations = self.requests_lib.put(
            url="http://{}:{}/generate".format(self.server_host, self.server_port),
            data=json.dumps(request),
            headers={"Content-Type": "application/json"},
        ).json()
        # we need to remove the original prompt as nemo always returns it
        outputs = [None] * len(generations['sentences'])
        for idx, generation in enumerate(generations['sentences']):
            # when the prompt starts from special tokens like bos, nemo will remove them,
            # so we need this hack to find where to start the cut
            begin_idx = 0
            while begin_idx < len(prompts[idx]) and not prompts[idx][begin_idx:].startswith(generation[:20]):
                begin_idx += 1
            outputs[idx] = {'generation': generation[(len(prompts[idx]) - begin_idx) :]}

        if remove_stop_phrases:
            for output in outputs:
                output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

        # TODO: return num_generated_tokens as well
        return outputs


class OpenAIModel(BaseModel):
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: str = '5000',
        model=None,
        base_url=None,
        api_key=None,
        max_retries: int = 3,
        initial_retry_delay: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from openai import OpenAI

        if model is None:
            model = os.getenv("NEMO_SKILLS_OPENAI_MODEL")
            if model is None:
                raise ValueError("model argument is required for OpenAI model.")

        if base_url is None:
            # if not provided, we assume it's served on host/port
            base_url = os.getenv("NEMO_SKILLS_OPENAI_BASE_URL", f"http://{host}:{port}/v1")

        if api_key is None:
            if base_url is not None and 'api.nvidia.com' in base_url:
                api_key = os.getenv("NVIDIA_API_KEY", api_key)
                if not api_key:
                    raise ValueError("NVIDIA_API_KEY is required for Nvidia-hosted models.")
            elif base_url is not None and 'api.openai.com' in base_url:
                api_key = os.getenv("OPENAI_API_KEY", api_key)
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is required for OpenAI models.")
            else:
                # assuming it's not used and setting a dummy value
                api_key = "dummy"

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        if self.model == "model":  # that's a placeholder, so trying to find real name
            self.model = self.get_model_name_from_server()

        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

    def batch_generate(
        self,
        prompts: list[str],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        top_logprobs: int | None = None,
        stop_phrases: list[str] | None = None,
    ) -> list[dict]:
        # only supported by the OpenAI endpoint!
        if stop_phrases is None:
            stop_phrases = []
        if top_k != 0:
            raise ValueError("`top_k` is not supported by OpenAI API, please set it to default value `0`.")

        # preparing the requests jsonl file
        with open("requests.jsonl", "wt", encoding='utf-8') as fout:
            for idx, prompt in enumerate(prompts):
                fout.write(
                    json.dumps(
                        {
                            "custom_id": f"{idx}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.model,
                                "messages": prompt,
                                "max_tokens": tokens_to_generate,
                                "temperature": temperature,
                                "top_p": top_p,
                                "presence_penalty": repetition_penalty,
                                "seed": random_seed,
                                "stop": stop_phrases,
                                "logprobs": top_logprobs is not None,
                                "top_logprobs": top_logprobs,
                            },
                        }
                    )
                    + "\n"
                )

        with open("requests.jsonl", "rb") as batch_file_handle:
            batch_file_id = self.client.files.create(file=batch_file_handle, purpose="batch").id

            metadata = self.client.batches.create(
                input_file_id=batch_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",  # the only supported value, but should finish faster
                metadata={"description": "batch job"},
            )

        return metadata

    def get_batch_results(self, batch_id):
        metadata = self.client.batches.retrieve(batch_id)
        outputs = None
        if metadata.status == 'completed' and metadata.output_file_id is not None:
            file_response = self.client.files.content(metadata.output_file_id)
            responses = file_response.text
            outputs = []
            for line in responses.split('\n')[:-1]:
                data = json.loads(line)
                outputs.append(
                    {
                        'custom_id': data['custom_id'],
                        'generation': data['response']['body']['choices'][0]['message']['content'],
                    }
                )
            outputs = sorted(outputs, key=lambda x: int(x['custom_id']))
            for output in outputs:
                output.pop('custom_id')

        return metadata, outputs

    def preprocess_request(self, request: dict):
        """OpenAI doesn't support top-k, so not making any changes here."""
        pass

    def _is_reasoning_model(self) -> bool:
        """Check if the current model is an OpenAI reasoning model (o1 series)."""
        reasoning_models = ['o1', 'o1-mini', 'o1-preview', 'o4-mini', 'o3-mini', 'o3']
        return any(reasoning_model in self.model.lower() for reasoning_model in reasoning_models)

    def _filter_reasoning_model_params(self, **kwargs) -> dict:
        """Filter out unsupported parameters for OpenAI reasoning models."""
        if not self._is_reasoning_model():
            return kwargs
        
        # Parameters unsupported by reasoning models
        unsupported_params = {
            'temperature', 'top_p', 'presence_penalty', 'frequency_penalty', 
            'logprobs', 'top_logprobs', 'logit_bias', 'max_tokens', 'tokens_to_generate', 'stop'
        }
        
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key not in unsupported_params:
                filtered_kwargs[key] = value
            else:
                LOG.warning(f"Parameter '{key}' is not supported by reasoning model {self.model}, ignoring")
        
        # For reasoning models, use max_completion_tokens instead of max_tokens/tokens_to_generate
        if 'tokens_to_generate' in kwargs:
            filtered_kwargs['max_completion_tokens'] = kwargs['tokens_to_generate']
        elif 'max_tokens' in kwargs:
            filtered_kwargs['max_completion_tokens'] = kwargs['max_tokens']
            
        return filtered_kwargs

    def _generate_single(
        self,
        prompt: dict,
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
        timeout: int | None = None,
        top_logprobs: int | None = None,
        stream: bool = False,
    ) -> dict | Stream:
        if top_k != 0:
            raise ValueError("`top_k` is not supported by OpenAI API, please set it to default value `0`.")
        if min_p > 0:
            raise ValueError("`min_p` is not supported by OpenAI API, please set it to default value `0`.")
        if top_logprobs is not None and top_logprobs > 1 and "integrate.api.nvidia.com" in str(self.client.base_url):
            raise ValueError("`top_logprobs` > 1 is not supported by Nvidia-hosted models.")
        if stream and top_logprobs is not None:
            raise ValueError("`top_logprobs` is not supported with stream=True")
        retry_count = 0
        retry_delay = self.initial_retry_delay

        while True:
            try:
                # Prepare API parameters
                api_params = {
                    'model': self.model,
                    'temperature': temperature,
                    'top_p': top_p,
                    'max_tokens': tokens_to_generate,
                    'presence_penalty': repetition_penalty,
                    'seed': random_seed,
                    'stop': stop_phrases,
                    'messages': prompt,
                    'logprobs': top_logprobs is not None,
                    'top_logprobs': top_logprobs,
                    'timeout': timeout,
                    'stream': stream,
                    'tokens_to_generate': tokens_to_generate,  # Used for reasoning model conversion
                }
                
                # Filter parameters for reasoning models
                filtered_params = self._filter_reasoning_model_params(**api_params)
                
                response = self.client.chat.completions.create(**filtered_params)
                break  # Success, exit the retry loop
            except openai.RateLimitError as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    LOG.error("Rate limit exceeded maximum retry attempts (%d). Giving up.", self.max_retries)
                    raise

                # Extract retry-after header if available, otherwise use exponential backoff
                retry_after = getattr(e, 'retry_after', None)
                if retry_after is not None and isinstance(retry_after, (int, float)):
                    wait_time = retry_after
                else:
                    wait_time = retry_delay
                    retry_delay *= 2  # Exponential backoff

                LOG.warning(
                    "Rate limit exceeded. Retrying in %.2f seconds... (Attempt %d/%d)",
                    wait_time,
                    retry_count,
                    self.max_retries,
                )
                time.sleep(wait_time)
            except openai.BadRequestError as e:
                # this likely only works for Nvidia-hosted models
                msg = e.body['detail']
                # expected message:
                # This model's maximum context length is N tokens.
                # However, you requested X tokens (Y in the messages, Z in the completion).
                # Please reduce the length of the messages or completion.
                if msg.startswith("This model's maximum context length is"):
                    numbers = re.findall(r"\d+", msg)
                    max_tokens = int(numbers[0]) - int(numbers[2])
                    LOG.warning("Reached max tokens! Reducing the number of tokens to generate to %d", max_tokens)
                    
                    # Prepare retry parameters with reduced max_tokens
                    retry_params = api_params.copy()
                    retry_params['max_tokens'] = max_tokens
                    retry_params['tokens_to_generate'] = max_tokens
                    
                    # Filter parameters for reasoning models
                    filtered_retry_params = self._filter_reasoning_model_params(**retry_params)
                    
                    response = self.client.chat.completions.create(**filtered_retry_params)
                else:
                    raise
            except AttributeError:
                LOG.error("Unexpected response from OpenAI API: %s", response)
                raise
            except Exception as e:
                LOG.error("Unexpected error during API call: %s", str(e))
                raise

        if stream:
            return self._stream_chunks(response)

        choice = response.choices[0]
        output = choice.message.content
        result = {'generation': output, 'num_generated_tokens': response.usage.completion_tokens}
        if choice.logprobs:
            result['logprobs'] = [tok.logprob for tok in choice.logprobs.content]
            result['tokens'] = [tok.token for tok in choice.logprobs.content]
            result['top_logprobs'] = []
            for token_logprob in choice.logprobs.content:
                logprob = {entry.token: entry.logprob for entry in token_logprob.top_logprobs}
                if token_logprob.token not in logprob:
                    logprob[token_logprob.token] = token_logprob.logprob
                result['top_logprobs'].append(logprob)

        return result

    def get_model_name_from_server(self):
        model_list = self.client.models.list()
        # TODO: this is a bit hacky, but will go away when we switch to a unified openai api for all models
        assert len(model_list.data) == 1, "Unexpected number of models returned by OpenAI API."
        model_name = model_list.data[0].id
        return model_name

    def _stream_chunks(self, response):
        """
        Helper generator that yields incremental chunks.
        """

        for chunk in response:
            cur_delta = chunk.choices[0].text

            if cur_delta:
                yield {"generation": cur_delta}

            stop_reason = getattr(chunk.choices[0], "stop_reason", None)
            if stop_reason and isinstance(stop_reason, str):
                yield {"generation": stop_reason}


class VLLMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TODO: move this to base model?
        self._tunnel = None
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel

            if '@' in self.ssh_server:
                ssh_username, ssh_server = self.ssh_server.split('@')
            else:
                ssh_server = self.ssh_server
                ssh_username = None

            self._tunnel = sshtunnel.SSHTunnelForwarder(
                (ssh_server, 22),
                ssh_username=ssh_username,
                ssh_pkey=self.ssh_key_path,
                remote_bind_address=(self.server_host, int(self.server_port)),
            )
            self._tunnel.start()
            # Use localhost with tunneled port for OpenAI client
            # This way all traffic to server_host:server_port goes through SSH tunnel
            self.server_host = '127.0.0.1'
            self.server_port = str(self._tunnel.local_bind_port)

        http_client = DefaultHttpxClient(
            limits=httpx.Limits(max_keepalive_connections=1500, max_connections=1500),
            transport=httpx.HTTPTransport(retries=3),
        )

        self.oai_client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"http://{self.server_host}:{self.server_port}/v1",
            timeout=None,
            http_client=http_client,
        )

        self.model_name_server = self.get_model_name_from_server()
        self.model = self.model_name_server

    def __del__(self):
        if self._tunnel:
            self._tunnel.stop()

    def _generate_single(
        self,
        prompt: str | dict,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stop_phrases: list[str] | None = None,
        stream: bool = False,
    ) -> dict | Stream:
        if isinstance(prompt, dict):
            raise NotImplementedError("TODO: need to add this support, but not implemented yet.")
        stop_phrases = stop_phrases or []

        if top_k == 0:
            top_k = -1

        response = self.oai_client.completions.create(
            model=self.model,
            prompt=[prompt],
            max_tokens=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            seed=random_seed,
            stop=stop_phrases,
            echo=False,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=top_logprobs,
            logit_bias=None,
            stream=stream,
            n=1,
            extra_body={
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "spaces_between_special_tokens": False,
            },
            timeout=timeout,
        )

        if stream:
            return self._stream_chunks(response)

        return self.parse_openai_response(response)

    @classmethod
    def parse_openai_response(cls, response: "openai.types.Completion") -> dict:
        assert not isinstance(response, list)
        assert len(response.choices) == 1
        choice = response.choices[0]
        output = choice.text
        # adding back stop words - somehow sometimes it returns token ids, so we do not handle those for now
        if choice.finish_reason == "stop":
            if hasattr(choice, "stop_reason") and isinstance(choice.stop_reason, str):
                output += choice.stop_reason
            # sglang has a little different api here
            if hasattr(choice, "matched_stop") and isinstance(choice.matched_stop, str):
                output += choice.matched_stop
        result = {'generation': output, 'num_generated_tokens': response.usage.completion_tokens}
        if choice.logprobs:
            result['logprobs'] = choice.logprobs.token_logprobs
            result['tokens'] = choice.logprobs.tokens
            result['top_logprobs'] = choice.logprobs.top_logprobs
        return result

    def get_model_name_from_server(self):
        model_list = self.oai_client.models.list()
        model_name = model_list.data[0].id
        return model_name

    def _stream_chunks(self, response):
        """
        Helper generator that yields incremental chunks.
        """

        emitted_so_far = []

        for chunk in response:
            cur_delta = chunk.choices[0].text
            emitted_so_far += [cur_delta]

            if cur_delta:
                yield {"generation": cur_delta}

            # vllm variant
            stop_reason = getattr(chunk.choices[0], "stop_reason", None)
            # sglang variant
            matched_stop = getattr(chunk.choices[0], "matched_stop", None)

            # vllm variant - emit stop_reason as is and finish
            if stop_reason and isinstance(stop_reason, str):
                yield {"generation": stop_reason}

            # sglang variant - emit only not-yet-sent part of matched_stop
            if matched_stop and isinstance(matched_stop, str):
                remaining = matched_stop
                # find the longest prefix of matched_stop that is already at
                # the end of what we've emitted.
                emitted_str = "".join(emitted_so_far)
                max_len = min(len(emitted_str), len(matched_stop))
                for i in range(max_len, 0, -1):
                    if emitted_str.endswith(matched_stop[:i]):
                        remaining = matched_stop[i:]
                        break

                if remaining:
                    yield {"generation": remaining}


class MegatronModel(BaseModel):
    # it's partially openai-compatible but not fully, so can't easily reuse the other class..
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TODO: duplication with vllm class, need to address this
        self._tunnel = None
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel

            if '@' in self.ssh_server:
                ssh_username, ssh_server = self.ssh_server.split('@')
            else:
                ssh_server = self.ssh_server
                ssh_username = None

            self._tunnel = sshtunnel.SSHTunnelForwarder(
                (ssh_server, 22),
                ssh_username=ssh_username,
                ssh_pkey=self.ssh_key_path,
                remote_bind_address=(self.server_host, int(self.server_port)),
            )
            self._tunnel.start()
            # Use localhost with tunneled port for OpenAI client
            # This way all traffic to server_host:server_port goes through SSH tunnel
            self.server_host = '127.0.0.1'
            self.server_port = str(self._tunnel.local_bind_port)

        http_client = DefaultHttpxClient(
            limits=httpx.Limits(max_keepalive_connections=1500, max_connections=1500),
            transport=httpx.HTTPTransport(retries=3),
        )

        self.oai_client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"http://{self.server_host}:{self.server_port}",  # note there is no /v1 here
            timeout=None,
            http_client=http_client,
        )

    def __del__(self):
        if self._tunnel:
            self._tunnel.stop()

    def _generate_single(
        self,
        prompt: str | dict,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stop_phrases: list[str] | None = None,
        stream: bool = False,
    ) -> dict | Stream:
        if isinstance(prompt, dict):
            raise NotImplementedError("Megatron server does not support OpenAI \"messages\" as prompt.")
        if stream is True:
            raise NotImplementedError("Megatron server does not support streaming.")
        if min_p > 0:
            raise NotImplementedError("Megatron server does not support min_p parameter.")
        if repetition_penalty != 1.0:
            raise NotImplementedError("Megatron server does not support repetition_penalty parameter.")
        if top_k != 0:
            raise NotImplementedError("Megatron server does not support top_k parameter.")

        stop_phrases = stop_phrases or []

        response = self.oai_client.completions.create(
            model="model",
            prompt=[prompt],
            max_tokens=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            seed=random_seed,
            stop=stop_phrases,
            echo=False,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=top_logprobs,
            logit_bias=None,
            stream=stream,
            n=1,
            timeout=timeout,
        )

        return self.parse_openai_response(response, top_logprobs=top_logprobs)

    def generate(
        self,
        prompts: list[str | dict],
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
    ) -> dict | Stream:
        # overriding generate to provide batched support as there is on inflight-batching
        if isinstance(prompts[0], dict):
            raise NotImplementedError("Megatron server does not support OpenAI \"messages\" as prompt.")
        if stream is True:
            raise NotImplementedError("Megatron server does not support streaming.")
        if min_p > 0:
            raise NotImplementedError("Megatron server does not support min_p parameter.")
        if repetition_penalty != 1.0:
            raise NotImplementedError("Megatron server does not support repetition_penalty parameter.")
        if top_k != 0:
            raise NotImplementedError("Megatron server does not support top_k parameter.")
        stop_phrases = stop_phrases or []

        if top_logprobs is None:
            top_logprobs = 0

        response = self.oai_client.completions.create(
            model="model",
            prompt=prompts,
            max_tokens=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            seed=random_seed,
            stop=stop_phrases,
            echo=False,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=top_logprobs,
            logit_bias=None,
            stream=stream,
            n=1,
            timeout=timeout,
        )

        outputs = self.parse_openai_response(response, batch=True, top_logprobs=top_logprobs)

        if remove_stop_phrases:
            for output in outputs:
                output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

        return outputs

    @classmethod
    def parse_openai_response(cls, response: "openai.types.Completion", batch: bool = False, top_logprobs: int | None = None) -> dict | list[dict]:
        """Parse OpenAI response to extract the generated text and other metadata.

        Args:
            response: The response from OpenAI API
            batch: Whether the response contains multiple generations (batch mode)

        Returns:
            A single dict with generation info or a list of dicts for batch mode
        """

        def process_choice(choice, top_logprobs: int | None = None):
            output = choice.text
            # adding back stop words - somehow sometimes it returns token ids, so we do not handle those for now
            if choice.finish_reason == "stop":
                if hasattr(choice, "stop_reason") and isinstance(choice.stop_reason, str):
                    output += choice.stop_reason
                # sglang has a little different api here
                if hasattr(choice, "matched_stop") and isinstance(choice.matched_stop, str):
                    output += choice.matched_stop

            result = {'generation': output, 'num_generated_tokens': -1}
            if choice.logprobs and choice.logprobs.tokens:  # logprobs is always populated, but empty if not requested
                if top_logprobs is not None and top_logprobs != 0:
                    result['logprobs'] = choice.logprobs.token_logprobs
                    result['tokens'] = choice.logprobs.tokens
                    result['top_logprobs'] = choice.logprobs.top_logprobs
                result['num_generated_tokens'] = len(choice.logprobs.tokens)
            return result

        if batch:
            return [process_choice(choice, top_logprobs) for choice in response.choices]
        else:
            assert not isinstance(response, list)
            assert len(response.choices) == 1
            return process_choice(response.choices[0], top_logprobs)


models = {
    'trtllm': TRTLLMModel,
    'nemo': NemoModel,
    'megatron': MegatronModel,
    'openai': OpenAIModel,
    'vllm': VLLMModel,
    'sglang': VLLMModel,  # interface is the same
}


def get_model(server_type, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = models[server_type.lower()]
    return model_class(**kwargs)
