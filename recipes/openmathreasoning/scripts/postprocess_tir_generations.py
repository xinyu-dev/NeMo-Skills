# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import glob
import json
import os
import re
from collections import Counter


def validate_code_execution(text, code_begin="```python", code_end="```", output_begin="```output", output_end="```", is_harmony_format=False):
    #NOTE: add parser for harmony format
    if is_harmony_format:
        # Parse sequentially to validate the order: code_begin -> code_end -> output_begin -> output_end (repeating >=0 times)
        # State: 0=expect code_begin, 1=expect code_end, 2=expect output_begin, 3=expect output_end
        state = 0
        position = 0
        complete_cycles = 0
        
        while position < len(text):
            if state == 0:  # Looking for code_begin
                next_pos = text.find(code_begin, position)
                if next_pos == -1:
                    break  # No more code blocks found
                position = next_pos + len(code_begin)
                state = 1
                
            elif state == 1:  # Looking for code_end
                next_pos = text.find(code_end, position)
                if next_pos == -1:
                    return False  # Unmatched code_begin
                position = next_pos + len(code_end)
                state = 2
                
            elif state == 2:  # Looking for output_begin
                next_pos = text.find(output_begin, position)
                if next_pos == -1:
                    return False  # Missing output_begin after code block
                position = next_pos + len(output_begin)
                state = 3
                
            elif state == 3:  # Looking for output_end
                next_pos = text.find(output_end, position)
                if next_pos == -1:
                    return False  # Unmatched output_begin
                position = next_pos + len(output_end)
                state = 0  # Complete cycle, look for next code_begin
                complete_cycles += 1
        
        # Valid if we have at least one complete cycle and ended in state 0 (not mid-cycle)
        return complete_cycles >= 0 and state == 0
    
    else:
        # in non-harmony format, we can rely on the `python\n` to parse the code blocks and output blocks.
        lines = text.split('\n')
        i = 0

        while i < len(lines):
            if lines[i] == code_begin:
                code_end_idx = -1
                for j in range(i + 1, len(lines)):
                    if lines[j] == code_end:
                        code_end_idx = j
                        break

                if code_end_idx == -1:
                    return False

                if code_end_idx + 1 >= len(lines) or lines[code_end_idx + 1] != output_begin:
                    return False

                output_end_idx = -1
                for j in range(code_end_idx + 2, len(lines)):
                    if lines[j] == output_end:
                        output_end_idx = j
                        break

                if output_end_idx == -1:
                    return False

                i = output_end_idx + 1
            else:
                i += 1

        return True



def cut_final_answer_part(output):
    final_answer_idx = output.find("**Final Answer**")
    if final_answer_idx == -1:
        return None

    boxed_idx = output.find("\\boxed{", final_answer_idx)
    if boxed_idx == -1:
        return None

    balance = 1
    end_idx = boxed_idx + 7
    while balance != 0 and end_idx < len(output):
        if output[end_idx] == "{":
            balance += 1
        elif output[end_idx] == "}":
            balance -= 1
        end_idx += 1

    return output[:end_idx]


def replace_code_tags(text, args):
    pattern = fr"{args.code_begin}(.*?)\n{args.code_end}"
    replacement = fr"{args.new_code_begin}\1\n{args.new_code_end}"
    processed_text = re.sub(pattern, replacement, text, flags=re.DOTALL)

    return processed_text


def filter_code_solution(sample, args):
    required_keys = ["predicted_answer", "generation", "problem"]
    for key in required_keys:
        if key not in sample:
            return "Key not found: " + key

    # Make some initial filtering to speed up the next llm judgement stage
    if args.code_begin not in sample["generation"]:
        return "No code blocks found"
    if not validate_code_execution(sample["generation"], args.code_begin.strip(), args.code_end.strip()):
        return "Incomplete code execution found"
    if "judgement" in sample and "judgement: no" in sample["judgement"].lower():
        return "Incorrect final answer"
    # if sample["generation"].find("\\boxed{") != -1 and sample["generation"].find("\\boxed{") < sample[
    #     "generation"
    # ].find(args.code_begin):
    #     return "Boxed before code"
    # if sample["generation"].find(sample["predicted_answer"]) != -1 and sample["generation"].find(
    #     sample["predicted_answer"]
    # ) < sample["generation"].find(args.code_begin):
    #     return "Predicted answer before code"

    # generation = cut_final_answer_part(sample["generation"])
    # if generation is None:
    #     return "Final answer not found"

    generation = sample["generation"]
    if args.new_code_begin and args.new_code_end:
        generation = replace_code_tags(generation, args)
    
    sample["generation"] = generation

    return "Accepted"


def preprocess_code_judge(args):
    cnt = Counter()
    # NOTE: collect failed samples
    failed_samples = []
    with open(args.output_file, "w") as fout:
        for input_file in glob.glob(args.input_files):
            with open(input_file, "r") as fin:
                for idx, line in enumerate(fin):
                    sample = json.loads(line)
                    filt_reason = filter_code_solution(sample, args)
                    cnt[filt_reason] += 1
                    cnt["Total"] += 1
                    if filt_reason == "Accepted":
                        sample["original_index"] = idx
                        fout.write(json.dumps(sample) + "\n")

                    # NOTE: add code to save the failed samples: 
                    else:
                        sample["original_index"] = idx
                        failed_samples.append(sample)

    # NOTE: added to save the failed samples to a separate file
    with open(args.output_file.replace(".jsonl", "_rejected.jsonl"), "w") as fout:
        for sample in failed_samples:
            fout.write(json.dumps(sample) + "\n")

    print("Filtered samples:")
    for key, value in cnt.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess code judge data")
    parser.add_argument(
        "--input_files", type=str, required=True, help="Input file, could be a pattern like output*.jsonl"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--code_begin", type=str, default="```python\n", help="Start of code block tag")
    parser.add_argument("--code_end", type=str, default="```\n", help="End of code block tag")
    parser.add_argument(
        "--new_code_begin", type=str, default=None,
        help="New start of code block tag, to replace the original one. If not specified, will not replace"
    )
    parser.add_argument(
        "--new_code_end", type=str, default=None,
        help="New end of code block tag, to replace the original one. If not specified, will not replace"
    )
    parser.add_argument("--output_begin", type=str, default="```output", help="Start of output block tag")
    parser.add_argument("--output_end", type=str, default="```", help="End of output block tag")
    parser.add_argument("--is_harmony_format", action="store_true", help="Whether the generation is in harmony text completionformat")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    preprocess_code_judge(args)
