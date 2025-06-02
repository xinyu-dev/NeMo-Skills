#!/usr/bin/env python3

"""
Test script to verify OpenAI code execution functionality.
"""

import os
import json
from nemo_skills.inference.server.code_execution_model import get_code_execution_model
from nemo_skills.code_execution.sandbox import get_sandbox

def test_openai_code_execution():
    """Test OpenAI model with code execution."""
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found. Please set it to test OpenAI models.")
        return
    
    print("Testing OpenAI code execution...")
    
    # Setup sandbox
    sandbox_config = {
        "sandbox_type": "local"
    }
    sandbox = get_sandbox(**sandbox_config)
    
    # Setup OpenAI model with code execution
    server_config = {
        "server_type": "openai",
        "model": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1"
    }
    
    code_execution_config = {
        "max_code_executions": 3,
        "code_execution_timeout": 10.0
    }
    
    try:
        llm = get_code_execution_model(
            sandbox=sandbox,
            code_execution=code_execution_config,
            **server_config
        )
        print("✓ Successfully created OpenAI code execution model")
        
        # Test with a simple math problem that requires code execution
        test_prompt = [
            {
                "role": "user", 
                "content": "Calculate 15 * 23 + 7 using Python code. You must use the exact format:\n<tool_call>\nprint(15 * 23 + 7)\n</tool_call>\nThen explain the result."
            }
        ]
        
        print("Testing generation with code execution...")
        
        result = llm.generate(
            prompts=[test_prompt],
            code_begin="<tool_call>\n",
            code_end="</tool_call>\n", 
            code_output_begin="<tool_output>\n",
            code_output_end="</tool_output>\n",
            code_output_format="text",
            tokens_to_generate=1000,
            temperature=0.0
        )
        
        print("✓ Generation completed successfully!")
        print(f"Result: {result[0]['generation']}")
        print(f"Code rounds executed: {result[0]['code_rounds_executed']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_openai_code_execution()
    if success:
        print("\n🎉 OpenAI code execution test passed!")
    else:
        print("\n❌ OpenAI code execution test failed!") 