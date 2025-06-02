#!/usr/bin/env python3

"""
Debug script to understand stop phrases in OpenAI TIR integration.
"""

from nemo_skills.prompt.utils import get_prompt

def debug_stop_phrases():
    """Debug the stop phrases configuration."""
    
    print("Debugging stop phrases configuration...")
    
    # Get TIR prompt configuration
    prompt = get_prompt('openmath/tir')  # TIR prompt config, no template
    
    print(f"Prompt template: {prompt.config.template}")
    print(f"Prompt stop phrases: {prompt.stop_phrases}")
    
    code_args = prompt.get_code_execution_args()
    print(f"Code execution args: {code_args}")
    
    # The issue is that code_end ('```\n') is being added as a stop phrase
    # This means the model stops generating as soon as it writes '```'
    # But we need it to complete the code block first
    
    print(f"\nThe problem:")
    print(f"- code_end ('{code_args['code_end']}') is added as a stop phrase")
    print(f"- This makes the model stop generating when it writes '```'")
    print(f"- But we need the model to complete the code block first")
    
    print(f"\nSolution:")
    print(f"- For OpenAI, we should NOT add code_end as a stop phrase")
    print(f"- Instead, let the model generate the complete code block")
    print(f"- Then detect when a complete code block is finished")

if __name__ == "__main__":
    debug_stop_phrases() 