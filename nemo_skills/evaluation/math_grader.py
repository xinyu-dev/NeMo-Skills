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

import glob
import json
import logging
import os
import re

import tqdm
from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify

from nemo_skills.utils import get_logger_name, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


def unroll_files(input_files):
    for manifest_pattern in input_files:
        for manifest in sorted(glob.glob(manifest_pattern, recursive=True)):
            yield manifest


def _additional_normalization(expr):
    # Remove % and \\% from the number
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match_gt = re.fullmatch(percentage_pattern, expr)
    if match_gt:
        expr = match_gt.group(1)
    # Remove . corresponding to the end of sentence
    expr = expr.rstrip(".\\")
    return expr


def math_equal(gt_answer, predicted_answer, take_modulo: int | None = None, **kwargs):
    if predicted_answer is None:
        return False

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)

    # if we are sure that gt is always integer
    if take_modulo is not None:
        gt_answer = int(gt_answer) % take_modulo
        try:
            predicted_answer = int(predicted_answer) % take_modulo
        except:
            predicted_answer = None
        # no need to simpy call in this case
        return predicted_answer == gt_answer

    # Try to compare as MCQ options
    mcq_options = "ABCDEFGHIJ"
    norm_gt_mcq = gt_answer.strip()

    is_mcq = re.fullmatch("|".join(mcq_options), norm_gt_mcq)
    parsed_gt = parse(gt_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    parsed_pred = parse(predicted_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    if is_mcq and verify(parsed_gt, parsed_pred):
        return verify(parsed_gt, parsed_pred)

    # Additional normalization step
    gt_answer = _additional_normalization(gt_answer)
    predicted_answer = _additional_normalization(predicted_answer)

    # Try literal comparison
    literal_pattern = r"[a-zA-Z ,]+|[0-9 ]+"
    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    is_literal = re.fullmatch(literal_pattern, normalized_gt) and re.fullmatch(literal_pattern, normalized_pred)
    is_normalized_equal = normalized_gt.replace(" ", "") == normalized_pred.replace(" ", "")

    if is_literal or is_normalized_equal:
        return is_normalized_equal

    # Fallback to symbolic comparison
    current_gt_answer = gt_answer
    current_predicted_answer = predicted_answer

    # math_verify.parse expects input to be in latex environment, e.g. $...$
    latex_env_search_pattern = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    if not re.search(latex_env_search_pattern, current_gt_answer, re.DOTALL):
        current_gt_answer = f"${current_gt_answer}$"
    if not re.search(latex_env_search_pattern, current_predicted_answer, re.DOTALL):
        current_predicted_answer = f"${current_predicted_answer}$"

    parsed_gt = parse(current_gt_answer, [LatexExtractionConfig()])
    parsed_pred = parse(current_predicted_answer, [LatexExtractionConfig()])

    return verify(parsed_gt, parsed_pred, **kwargs)


def batch_evaluate_results(
    input_files: list[str],
    numeric_precision=15,
    timeout=10,
    take_modulo=None,
    use_predicted_answer_key: bool = False,
    extract_from_boxed: bool = True,
    extract_regex: str = r"The final answer is (.+)$",
):
    for input_file in tqdm.tqdm(unroll_files(input_files), desc="Processing files"):
        # assume that input_file is small enough to entirely fit in the memory
        input_data = []
        with open(input_file, "rt", encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)

        with open(input_file, "rt", encoding="utf-8") as fin:
            for file_line in tqdm.tqdm(fin, total=num_lines, desc=f"Evaluating {os.path.basename(input_file)}"):
                line_dict = json.loads(file_line)
                if not line_dict:  # can be empty for incomplete generations
                    input_data.append({})
                    continue

                if not use_predicted_answer_key:
                    line_dict["predicted_answer"] = extract_answer(
                        line_dict["generation"],
                        extract_from_boxed=extract_from_boxed,
                        extract_regex=extract_regex,
                    )
                else:
                    if "predicted_answer" not in line_dict:
                        raise ValueError(
                            "predicted_answer key not found in the line_dict. "
                            "Set use_predicted_answer_key=False to re-extract"
                        )

                gt_answer = line_dict["expected_answer"]
                predicted_answer = line_dict["predicted_answer"]

                line_dict["symbolic_correct"] = math_equal(
                    gt_answer,
                    predicted_answer,
                    take_modulo=take_modulo,
                    numeric_precision=numeric_precision,
                    timeout_seconds=timeout,
                )
                input_data.append(line_dict)
        with open(input_file, "wt", encoding="utf-8", buffering=1) as fout:
            for line_dict in input_data:
                fout.write(json.dumps(line_dict) + "\n")


def extract_answer(string: str, extract_from_boxed: bool = True, extract_regex: str = r"The final answer is (.+)$"):
    """Extract everything from 'Answer: ' to the end, or return whole string if 'Answer: ' not found"""
    # Look for "Answer: " pattern (case-insensitive)

    # Attempt 1: try matching the harmony final answer pattern
    # this must be placed first, since gpt-oss models like to mumble "final answer" in its planning process, 
    # even though it hasn't generated the final answer yet. 
    # must use escape to avoid regex injection!!
    answer_pattern = re.escape("<|start|>assistant<|channel|>final<|message|>")
    matches = list(re.finditer(answer_pattern, string, re.IGNORECASE))
    match = matches[-1] if matches else None
    if match:
        # Extract everything from the start of "Answer: " to the end
        return string[match.end():].strip()

    # Attempt 2:  try matching the final answer pattern
    answer_pattern = r"Final answer:\s*"
    match = re.search(answer_pattern, string, re.IGNORECASE)
    if match:
        # Extract everything from the start of "Answer: " to the end
        return string[match.start():].strip()
    else:
        # If "Answer: " not found, return the whole input string or None
        # if returnning None, then this trace will get filtered out as incorrect
        # return string.strip()
        return None
