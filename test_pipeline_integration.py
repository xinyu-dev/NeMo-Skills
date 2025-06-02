#!/usr/bin/env python3

"""
Test script to verify that the pipeline correctly passes code execution markers.
"""

import os
import tempfile
import json
from pathlib import Path
from nemo_skills.inference.generate import GenerationTask, GenerateSolutionsConfig

def test_code_execution_markers():
    """Test that code execution markers are correctly passed to the model."""
    
    # Create a temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_data = {"problem": "Calculate 2 + 2", "expected_answer": "4"}
        f.write(json.dumps(test_data) + '\n')
        input_file = f.name
    
    # Create a temporary output file
    output_file = tempfile.mktemp(suffix='.jsonl')
    
    try:
        # Create config for OpenAI with code execution
        config = GenerateSolutionsConfig(
            output_file=output_file,
            input_file=input_file,
            prompt_config="generic/math",
            prompt_template=None,  # This simulates OpenAI usage
            code_execution=True,
            server={
                "server_type": "openai",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1"
            },
            sandbox={"sandbox_type": "local"},
            dry_run=True,  # Don't actually run generation
            max_samples=1,
            extra_stop_phrases=[]  # Fix the OmegaConf issue
        )
        
        # Create generation task
        task = GenerationTask(config)
        
        # Setup components
        task.llm = task.setup_llm()
        task.prompt = task.setup_prompt()
        
        # Load and preprocess data
        data = task.load_data()
        data = task.preprocess_data(data)
        
        # Test that prompt has the get_code_execution_args method
        code_args = task.prompt.get_code_execution_args()
        print("✓ Code execution arguments from prompt:")
        for key, value in code_args.items():
            print(f"  {key}: {repr(value)}")
        
        # Verify the expected markers for OpenAI (no template)
        expected_markers = {
            "code_begin": "```python\n",
            "code_end": "```\n", 
            "code_output_begin": "```output\n",
            "code_output_end": "```\n",
            "code_output_format": "qwen"
        }
        
        for key, expected_value in expected_markers.items():
            if code_args[key] == expected_value:
                print(f"✓ {key} matches expected value")
            else:
                print(f"✗ {key} mismatch: got {repr(code_args[key])}, expected {repr(expected_value)}")
                return False
        
        print("\n✓ All code execution markers are correctly configured!")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(output_file):
            os.unlink(output_file)

if __name__ == "__main__":
    success = test_code_execution_markers()
    if success:
        print("\n🎉 Pipeline integration test passed!")
    else:
        print("\n❌ Pipeline integration test failed!") 