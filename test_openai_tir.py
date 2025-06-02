#!/usr/bin/env python3

"""
Test OpenAI TIR integration with the corrected configuration.
"""

import os
from nemo_skills.inference.server.code_execution_model import get_code_execution_model
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.prompt.utils import get_prompt

def test_openai_tir():
    """Test OpenAI TIR integration with a simple math problem."""
    
    print("Testing OpenAI TIR integration...")
    
    # Setup sandbox
    sandbox_config = {"sandbox_type": "local"}
    sandbox = get_sandbox(**sandbox_config)
    
    # Setup OpenAI model with code execution
    server_config = {
        "server_type": "openai",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1"
    }
    
    # Get TIR prompt configuration
    prompt = get_prompt('openmath/tir')  # TIR prompt config
    code_args = prompt.get_code_execution_args()
    
    print(f"Using prompt config: openmath/tir")
    print(f"Code execution markers: {code_args}")
    
    try:
        # Create code execution model
        llm = get_code_execution_model(sandbox=sandbox, **server_config)
        
        # Test with a simple math problem
        test_problem = "Calculate the sum of the first 5 prime numbers."
        
        # Fill the prompt with total_code_executions
        filled_prompt = prompt.fill({
            'problem': test_problem,
            'total_code_executions': 3
        })
        
        print(f"\nTest problem: {test_problem}")
        print(f"Filled prompt: {filled_prompt}")
        
        # Generate solution with code execution
        print("\nGenerating solution with OpenAI + code execution...")
        
        generation_params = {
            "prompts": [filled_prompt],
            "tokens_to_generate": 500,  # Increased to allow complete code blocks
            "temperature": 0.1,
            **code_args
        }
        
        outputs = llm.generate(**generation_params)
        
        if outputs and len(outputs) > 0:
            result = outputs[0]
            print(f"\n=== GENERATION RESULT ===")
            print(f"Generation: {result['generation']}")
            print(f"Code rounds executed: {result.get('code_rounds_executed', 'N/A')}")
            print(f"Generation time: {result.get('generation_time', 'N/A')}s")
            print(f"Code execution time: {result.get('code_execution_time', 'N/A')}s")
            
            # Check if code was executed
            if result.get('code_rounds_executed', 0) > 0:
                print("✅ Code execution successful!")
                
                # Check if the generation contains expected patterns
                generation = result['generation']
                if '```python' in generation and '```output' in generation:
                    print("✅ Generated solution contains code blocks and outputs!")
                else:
                    print("⚠️  Generated solution missing expected code/output blocks")
                    
            else:
                print("⚠️  No code was executed")
                
        else:
            print("❌ No output generated")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_openai_tir()
    if success:
        print("\n🎯 OpenAI TIR integration test completed!")
    else:
        print("\n❌ OpenAI TIR integration test failed!") 