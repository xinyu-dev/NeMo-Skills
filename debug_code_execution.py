#!/usr/bin/env python3

"""
Debug script to check code execution configuration.
"""

def debug_prompt_config():
    """Check what code execution args are being used."""
    
    from nemo_skills.prompt.utils import get_prompt
    
    # Load your TIR prompt
    prompt = get_prompt("nemo_skills/prompt/config/openmath/tir")
    
    print("=== PROMPT CONFIGURATION ===")
    print(f"Template: {prompt.config.template}")
    print(f"System: {repr(prompt.config.system)}")
    print(f"User prompt (first 200 chars): {repr(prompt.config.user[:200])}...")
    print()
    
    print("=== CODE EXECUTION ARGS ===")
    code_args = prompt.get_code_execution_args()
    for key, value in code_args.items():
        print(f"{key}: {repr(value)}")
    print()
    
    print("=== STOP PHRASES ===")
    stop_phrases = prompt.stop_phrases
    print(f"Stop phrases: {stop_phrases}")
    print()
    
    # Test with a sample problem
    print("=== SAMPLE PROMPT FILL ===")
    sample_data = {
        "problem": "What is 2 + 2?",
        "total_code_executions": 3
    }
    
    filled_prompt = prompt.fill(sample_data)
    print(f"Filled prompt type: {type(filled_prompt)}")
    if isinstance(filled_prompt, list):
        for i, msg in enumerate(filled_prompt):
            print(f"Message {i}: {msg}")
    else:
        print(f"Filled prompt: {repr(filled_prompt)}")
    
    return code_args

if __name__ == "__main__":
    debug_prompt_config() 