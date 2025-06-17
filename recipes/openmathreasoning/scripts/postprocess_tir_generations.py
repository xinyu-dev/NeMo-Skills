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
from collections import Counter, defaultdict


def validate_code_execution(text, code_begin="```python", code_end="```"):
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

            if code_end_idx + 1 >= len(lines) or lines[code_end_idx + 1] != "```output":
                return False

            output_end_idx = -1
            for j in range(code_end_idx + 2, len(lines)):
                if lines[j] == "```":
                    output_end_idx = j
                    break

            if output_end_idx == -1:
                return False

            i = output_end_idx + 1
        else:
            i += 1

    return True


def cut_final_answer_part(output):
    # final_answer_idx = output.find("**Final Answer**")
    # Try multiple patterns for final answer
    patterns = ["**Final Answer**", "Final Answer:", "Final Answer"]
    final_answer_idx = -1
    
    for pattern in patterns:
        final_answer_idx = output.find(pattern)
        if final_answer_idx != -1:
            break
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


def filter_code_solution(sample, args, rejected_samples):
    required_keys = ["predicted_answer", "generation", "problem"]
    for key in required_keys:
        if key not in sample:
            rejected_samples["Key not found"].append(sample)
            return "Key not found: " + key

    # Make some initial filtering to speed up the next llm judgement stage
    if args.code_begin not in sample["generation"]:
        rejected_samples["No code blocks found"].append(sample)
        return "No code blocks found"
    if not validate_code_execution(sample["generation"], args.code_begin.strip(), args.code_end.strip()):
        rejected_samples["Incomplete code execution found"].append(sample)
        return "Incomplete code execution found"
    if "judgement" in sample and "judgement: no" in sample["judgement"].lower():
        rejected_samples["Incorrect final answer"].append(sample)
        return "Incorrect final answer"
    if sample["generation"].find("\\boxed{") != -1 and sample["generation"].find("\\boxed{") < sample[
        "generation"
    ].find(args.code_begin):
        rejected_samples["Boxed before code"].append(sample)
        return "Boxed before code"
    # the original filtering simply detects the answer before the code block, which results in a lot of false positives cases. 
    # if sample["generation"].find(sample["predicted_answer"]) != -1 and sample["generation"].find(
    #     sample["predicted_answer"]
    # ) < sample["generation"].find(args.code_begin):
    #     rejected_samples["Predicted answer before code"].append(sample)
    #     return "Predicted answer before code"

    generation = cut_final_answer_part(sample["generation"])
    if generation is None:
        rejected_samples["Final answer not found"].append(sample)
        return "Final answer not found"

    if args.new_code_begin and args.new_code_end:
        generation = replace_code_tags(generation, args)
    
    sample["generation"] = generation

    return "Accepted"


def preprocess_code_judge(args):
    cnt = Counter()
    rejected_samples = defaultdict(list)
    
    with open(args.output_file, "w") as fout:
        for input_file in glob.glob(args.input_files):
            with open(input_file, "r") as fin:
                for idx, line in enumerate(fin):
                    sample = json.loads(line)
                    filt_reason = filter_code_solution(sample, args, rejected_samples)
                    cnt[filt_reason] += 1
                    cnt["Total"] += 1
                    if filt_reason == "Accepted":
                        sample["original_index"] = idx
                        fout.write(json.dumps(sample) + "\n")

    # Save rejected samples to separate JSON files
    output_dir = os.path.dirname(args.output_file)
    
    for reason, samples in rejected_samples.items():
        if samples:  # Only create file if there are samples
            filename = f"rejected_output_{reason.lower().replace(' ', '_')}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(samples, f, indent=2)
            print(f"Saved {len(samples)} rejected samples to {filepath}")

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
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    preprocess_code_judge(args)
