#!/usr/bin/env python3

"""
Debug script to check the new output format.
"""

def debug_new_output():
    """Debug the new output."""
    
    # This is what the model generated with the new stop phrase
    output = """The first 5 prime numbers are 2, 3, 5, 7, and 11. To find their sum, we can add these numbers together.

Let's calculate the sum using Python code to ensure accuracy.

```python
# List of the first 5 prime numbers
prime_numbers = [2, 3, 5, 7, 11]

# Calculate the sum of the prime numbers
sum_of_primes = sum(prime_numbers)
sum_of_primes"""

    code_begin = "```python\n"
    code_end = "```\n"
    
    print("Debugging new output...")
    print(f"Output: {repr(output)}")
    print(f"Output ends with: {repr(output[-20:])}")
    print(f"Does output end with code_end? {output.endswith(code_end)}")
    print(f"Code_end: {repr(code_end)}")
    
    # Check if the output contains a complete code block
    last_code_begin = output.rfind(code_begin)
    print(f"Last code_begin position: {last_code_begin}")
    
    if last_code_begin != -1:
        # Look for code_end after the last code_begin
        code_end_pos = output.find(code_end, last_code_begin + len(code_begin))
        print(f"Code_end position after last code_begin: {code_end_pos}")
        
        if code_end_pos == -1:
            print("❌ Code block is incomplete - missing closing ```")
            print("The model stopped before completing the code block")
            print("This suggests the stop phrase ```\\n\\n is still stopping too early")
        else:
            print("✅ Code block is complete")
            code_block = output[last_code_begin:code_end_pos + len(code_end)]
            print(f"Complete code block: {repr(code_block)}")

if __name__ == "__main__":
    debug_new_output() 