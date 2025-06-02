#!/usr/bin/env python3

"""
Simple test to verify that code execution markers work correctly.
"""

from nemo_skills.prompt.utils import get_prompt

def test_code_execution_markers():
    """Test that code execution markers are correctly retrieved from prompts."""
    
    print("Testing code execution markers...")
    
    # Test 1: With template (like local models)
    print("\n1. Testing with template (local models):")
    prompt_with_template = get_prompt('generic/math', 'default-base')  # Uses <llm-code> format
    code_args_template = prompt_with_template.get_code_execution_args()
    
    print("  Code execution arguments with template:")
    for key, value in code_args_template.items():
        print(f"    {key}: {repr(value)}")
    
    # Test 2: Without template (like OpenAI)
    print("\n2. Testing without template (OpenAI models):")
    prompt_no_template = get_prompt('generic/math')  # No template = OpenAI format
    code_args_no_template = prompt_no_template.get_code_execution_args()
    
    print("  Code execution arguments without template:")
    for key, value in code_args_no_template.items():
        print(f"    {key}: {repr(value)}")
    
    # Verify the expected markers for OpenAI (no template)
    expected_openai_markers = {
        "code_begin": "```python\n",
        "code_end": "```\n", 
        "code_output_begin": "```output\n",
        "code_output_end": "```\n",
        "code_output_format": "qwen"
    }
    
    print("\n3. Verifying OpenAI markers:")
    all_correct = True
    for key, expected_value in expected_openai_markers.items():
        actual_value = code_args_no_template[key]
        if actual_value == expected_value:
            print(f"  ✓ {key}: {repr(actual_value)}")
        else:
            print(f"  ✗ {key}: got {repr(actual_value)}, expected {repr(expected_value)}")
            all_correct = False
    
    # Verify template markers are different (should use template-specific markers)
    print("\n4. Verifying template markers are different from OpenAI:")
    template_different = False
    for key in expected_openai_markers:
        if code_args_template[key] != code_args_no_template[key]:
            template_different = True
            print(f"  ✓ {key} differs: template={repr(code_args_template[key])}, openai={repr(code_args_no_template[key])}")
    
    if not template_different:
        print("  ✗ Template and OpenAI markers are the same (unexpected)")
        all_correct = False
    
    return all_correct

if __name__ == "__main__":
    success = test_code_execution_markers()
    if success:
        print("\n🎉 Code execution markers test passed!")
        print("\nThis means:")
        print("  - OpenAI models will use ```python``` format for code")
        print("  - Local models will use template-specific formats")
        print("  - The pipeline should now correctly pass these markers to the code execution wrapper")
    else:
        print("\n❌ Code execution markers test failed!") 