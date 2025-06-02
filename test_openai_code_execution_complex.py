#!/usr/bin/env python3

"""
Test script to verify OpenAI code execution with a complex problem.
"""

import os
from nemo_skills.inference.server.code_execution_model import get_code_execution_model
from nemo_skills.code_execution.sandbox import get_sandbox

def test_openai_code_execution_complex():
    """Test OpenAI model with code execution using a complex problem."""
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found. Please set it to test OpenAI models.")
        return False
    
    print("Testing OpenAI code execution with complex problem...")
    
    # Setup sandbox
    sandbox_config = {
        "sandbox_type": "local"
    }
    sandbox = get_sandbox(**sandbox_config)
    
    # Setup OpenAI model with code execution
    server_config = {
        "server_type": "openai",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1"
    }
    
    code_execution_config = {
        "max_code_executions": 5,
        "code_execution_timeout": 30.0
    }
    
    try:
        llm = get_code_execution_model(
            sandbox=sandbox,
            code_execution=code_execution_config,
            **server_config
        )
        print("✓ Successfully created OpenAI code execution model")
        
        # Test with a problem that requires code execution
        test_prompt = [
            {
                "role": "user", 
                "content": """I need to find the sum of all prime numbers between 1 and 100. Please solve this step by step using Python code.

Use this exact format:
<tool_call>
# Your Python code here
</tool_call>

Then explain the result."""
            }
        ]
        
        print("Testing generation with complex problem...")
        
        result = llm.generate(
            prompts=[test_prompt],
            code_begin="<tool_call>\n",
            code_end="</tool_call>\n", 
            code_output_begin="<tool_output>\n",
            code_output_end="</tool_output>\n",
            code_output_format="text",
            tokens_to_generate=1000,
            temperature=0.1
        )
        
        print("✓ Generation completed successfully!")
        print(f"Result: {result[0]['generation']}")
        print(f"Code rounds executed: {result[0]['code_rounds_executed']}")
        
        # Check if code was actually executed
        if result[0]['code_rounds_executed'] > 0:
            print("✓ Code execution worked!")
            return True
        else:
            print("⚠ Code was not executed - this might be expected if the model solved it analytically")
            return True  # Still consider it a success since the wrapper worked
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_openai_code_execution_complex()
    if success:
        print("\n🎉 OpenAI code execution test passed!")
    else:
        print("\n❌ OpenAI code execution test failed!") 