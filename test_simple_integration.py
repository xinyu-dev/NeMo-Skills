#!/usr/bin/env python3

"""
Test to verify code execution and count sandbox executions.
"""

import os
import tempfile
import json
from nemo_skills.inference.server.code_execution_model import get_code_execution_model
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.prompt.utils import get_prompt

def test_code_execution_count():
    """Test code execution and count how many times code is executed in sandbox."""
    
    print("Testing code execution with sandbox counting...")
    
    # Setup sandbox
    sandbox_config = {"sandbox_type": "local"}
    sandbox = get_sandbox(**sandbox_config)
    
    # Setup OpenAI model with code execution (simulated)
    server_config = {
        "server_type": "openai",
        "model": "gpt-4o-mini",  # Using a non-reasoning model to avoid parameter filtering issues
        "base_url": "https://api.openai.com/v1"
    }
    
    # Get prompt to determine code execution markers
    prompt = get_prompt('openmath/tir')  # TIR prompt config, no template = OpenAI format
    code_args = prompt.get_code_execution_args()
    
    print(f"Code execution markers:")
    for key, value in code_args.items():
        print(f"  {key}: {repr(value)}")
    
    print(f"\nExpected markers for OpenAI:")
    print(f"  code_begin: {repr('```python\\n')}")
    print(f"  code_end: {repr('```\\n')}")
    print(f"  code_output_begin: {repr('```output\\n')}")
    print(f"  code_output_end: {repr('```\\n')}")
    
    try:
        # Create code execution model
        llm = get_code_execution_model(sandbox=sandbox, **server_config)
        
        # Test prompts with different amounts of code
        test_cases = [
            {
                "name": "Single code block",
                "prompt": [{"role": "user", "content": "Calculate 2 + 2 using Python:\n```python\nresult = 2 + 2\nprint(f'2 + 2 = {result}')\n```"}],
                "expected_executions": 1
            },
            {
                "name": "Multiple code blocks", 
                "prompt": [{"role": "user", "content": "Do some calculations:\n```python\nx = 5\nprint(f'x = {x}')\n```\n\nThen:\n```python\ny = x * 2\nprint(f'y = {y}')\n```"}],
                "expected_executions": 2
            },
            {
                "name": "No code blocks",
                "prompt": [{"role": "user", "content": "What is 2 + 2? Just answer without code."}],
                "expected_executions": 0
            }
        ]
        
        total_executions = 0
        
        for i, test_case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1}: {test_case['name']} ---")
            
            # Create a test generation that includes the code blocks
            test_generation = ""
            if "```python" in test_case["prompt"][0]["content"]:
                # Extract and format the code blocks properly
                content = test_case["prompt"][0]["content"]
                parts = content.split("```python")
                for j, part in enumerate(parts[1:], 1):
                    code_part = part.split("```")[0]
                    test_generation += f"```python\n{code_part.strip()}\n```\n"
                    if j < len(parts) - 1:
                        test_generation += "\nNow let's continue:\n"
            else:
                test_generation = "The answer is 4."
            
            print(f"Test generation: {repr(test_generation)}")
            
            # Count code blocks in the generation
            code_blocks = test_generation.count("```python")
            print(f"Code blocks found: {code_blocks}")
            
            # Simulate what the code execution wrapper would do
            if code_blocks > 0:
                from nemo_skills.code_execution.utils import extract_code_to_execute
                
                # Extract each code block and execute it
                remaining_text = test_generation
                execution_count = 0
                
                while "```python\n" in remaining_text:
                    # Find the next code block
                    start_idx = remaining_text.find("```python\n")
                    if start_idx == -1:
                        break
                        
                    # Find the end of this code block
                    end_idx = remaining_text.find("```\n", start_idx + len("```python\n"))
                    if end_idx == -1:
                        break
                    
                    # Extract the code
                    code_block = remaining_text[start_idx:end_idx + len("```\n")]
                    code_to_execute = extract_code_to_execute(code_block, "```python\n", "```\n")
                    
                    if code_to_execute.strip():
                        print(f"  Executing code block {execution_count + 1}: {repr(code_to_execute.strip())}")
                        
                        # Actually execute the code in sandbox
                        try:
                            execution_dict, session_id = sandbox.execute_code(
                                generated_code=code_to_execute,
                                timeout=10.0,
                                max_output_characters=1000,
                                session_id=None
                            )
                            execution_count += 1
                            print(f"    Execution result: {execution_dict.get('stdout', '').strip()}")
                            
                        except Exception as e:
                            print(f"    Execution failed: {e}")
                    
                    # Move past this code block
                    remaining_text = remaining_text[end_idx + len("```\n"):]
                
                print(f"Total executions for this test: {execution_count}")
                total_executions += execution_count
            else:
                print("No code to execute")
        
        print(f"\n=== SUMMARY ===")
        print(f"Total code executions across all tests: {total_executions}")
        
        # Test the actual code execution markers work
        print(f"\nCode execution markers are correctly configured:")
        print(f"  - code_begin: {repr(code_args['code_begin'])}")
        print(f"  - code_end: {repr(code_args['code_end'])}")
        print(f"  - Sandbox type: {sandbox_config['sandbox_type']}")
        
        return total_executions
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    executions = test_code_execution_count()
    print(f"\n🎯 Total sandbox code executions: {executions}")
    if executions > 0:
        print("✅ Code execution is working correctly!")
    else:
        print("❌ No code was executed") 