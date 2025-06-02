#!/usr/bin/env python3

"""
Debug script to check the actual output from the test.
"""

def debug_actual_output():
    """Debug the actual output."""
    
    # This is what the model actually generated in the test
    output = """The first 5 prime numbers are 2, 3, 5, 7, and 11. To find their sum, we can add these numbers together.

Let's calculate the sum using Python code to ensure accuracy:

```python
# List of the first 5 prime numbers
prime_numbers = [2, 3, 5, 7, 11]

# Calculate the sum
sum_of_primes = sum(prime_numbers)
sum_of_primes"""

    code_begin = "```python\n"
    code_end = "```\n"
    
    print("Debugging actual output...")
    print(f"Output: {repr(output)}")
    print(f"Output ends with: {repr(output[-20:])}")
    print(f"Does output end with code_end? {output.endswith(code_end)}")
    print(f"Code_end: {repr(code_end)}")
    
    # The issue: the model stopped before writing the closing ```
    # This is because ``` is used as a stop phrase, but it stops too early
    
    print(f"\nThe problem:")
    print(f"- The model started writing a code block with {repr(code_begin)}")
    print(f"- But the stop phrase {repr(code_end)} made it stop before completing the block")
    print(f"- We need a different approach")
    
    print(f"\nPossible solutions:")
    print(f"1. Use a more specific stop phrase that only triggers after complete code blocks")
    print(f"2. Use a different marker system")
    print(f"3. Let the model generate more and detect complete blocks post-generation")

if __name__ == "__main__":
    debug_actual_output() 