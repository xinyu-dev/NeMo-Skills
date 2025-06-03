#!/usr/bin/env python3

import json
import glob
import os
import re
from os import path

def read_jsonl_file(file_path: str):
    instances = []
    with open(file_path, "r") as f:
        for line in f:
            instance = json.loads(line)
            instances.append(instance)
    return instances

def is_correct_judgement(judgement):
    """Simple check for correct judgement"""
    return judgement and "Yes" in judgement

def is_valid_summary(reasoning_instance, summary_instance):
    """Identify if the summary is valid for the reasoning solution"""
    # If both the reasoning solution and the summary are judged correct, then the summary is valid
    if is_correct_judgement(reasoning_instance["judgement"]) and is_correct_judgement(summary_instance["judgement"]):
        return True
    # Otherwise check for the surface form to ensure that the summary has the same answer
    return (reasoning_instance["predicted_answer"] == summary_instance["predicted_answer"])

def trim_reasoning_generation(reasoning_generation, start_tag="<think>", end_tag="</think>", strict_end_tag=False):    
    """Trim the thinking part of the original reasoning generation till the step with the rightmost boxed entry"""
    
    # Find the start and end tags. If either is not found, return None
    start_tag_position = reasoning_generation.find(start_tag)
    if start_tag_position == -1:
        return None

    end_tag_position = reasoning_generation.find(end_tag)
    if end_tag_position == -1:
        if strict_end_tag:
            return None
        else:
            reasoning_generation = reasoning_generation + end_tag
            reasoning_trace = reasoning_generation
    else:
        reasoning_trace = reasoning_generation[:end_tag_position + len(end_tag)]

    return reasoning_trace

def debug_merge():
    reasoning_file = "/home/ubuntu/NeMo-Skills/workspace/openmathreasoning-demo/solution-sdg-omr/step-4-postprocess-tir/postprocessed_output.jsonl"
    summary_dir = "/home/ubuntu/NeMo-Skills/workspace/openmathreasoning-demo/solution-sdg-omr/step-6-judge-new-summaries"
    
    print(f"Reasoning file: {reasoning_file}")
    print(f"Summary directory: {summary_dir}")
    print(f"Summary directory exists: {path.exists(summary_dir)}")
    
    # Read reasoning instances
    reasoning_instances = read_jsonl_file(reasoning_file)
    print(f"Number of reasoning instances: {len(reasoning_instances)}")
    
    # Find summary files
    summary_files = glob.glob(path.join(summary_dir, "*.jsonl"))
    print(f"Found summary files: {summary_files}")
    
    # Read summary instances
    list_of_summary_instances = []
    for summary_file in summary_files:
        summary_instances = read_jsonl_file(summary_file)
        print(f"Summary file {summary_file}: {len(summary_instances)} instances")
        list_of_summary_instances.append(summary_instances)
    
    # Filter by length
    valid_summary_lists = [summary_instances for summary_instances in list_of_summary_instances 
                          if len(reasoning_instances) == len(summary_instances)]
    print(f"Valid summary lists (matching length): {len(valid_summary_lists)}")
    
    if len(valid_summary_lists) > 0:
        # Check first few instances for validation
        all_summaries = list(zip(*valid_summary_lists))
        print(f"Number of summary groups: {len(all_summaries)}")
        
        # Test the actual merge logic on first few instances
        formatted_count = 0
        for i, (reasoning_instance, summaries_for_reasoning_instance) in enumerate(zip(reasoning_instances[:3], all_summaries[:3])):
            print(f"\n=== Instance {i} ===")
            
            # Check if reasoning generation has the required tags
            generation = reasoning_instance["generation"]
            print(f"Generation length: {len(generation)}")
            print(f"Has <think>: {'<think>' in generation}")
            print(f"Has </think>: {'</think>' in generation}")
            
            # Step 1 - Trim the reasoning generation
            trimmed_reasoning_trace = trim_reasoning_generation(generation, "<think>", "</think>", strict_end_tag=False)
            print(f"Trimmed reasoning trace: {trimmed_reasoning_trace is not None}")
            if trimmed_reasoning_trace:
                print(f"Trimmed length: {len(trimmed_reasoning_trace)}")
            
            # If the reasoning generation is not trimmed, skip this instance
            if trimmed_reasoning_trace is None:
                print("Skipping: Could not trim reasoning generation")
                continue
            
            # Check valid summaries
            valid_summaries = []
            for j, summary_instance in enumerate(summaries_for_reasoning_instance):
                is_valid = is_valid_summary(reasoning_instance, summary_instance)
                print(f"  Summary {j} valid: {is_valid}")
                if is_valid:
                    valid_summaries.append(summary_instance)
            
            print(f"Valid summaries count: {len(valid_summaries)}")
            
            if len(valid_summaries) > 0:
                formatted_count += 1
                print(f"Would format this instance!")
            else:
                print(f"Skipping: No valid summaries")
        
        print(f"\nTotal instances that would be formatted: {formatted_count}")

if __name__ == "__main__":
    debug_merge() 