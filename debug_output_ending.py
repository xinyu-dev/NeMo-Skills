#!/usr/bin/env python3

"""
Debug script to check output ending and code detection.
"""

def debug_output_ending():
    """Debug what the output looks like."""
    
    # Sample output from the test
    output = """The first 5 prime numbers are 2, 3, 5, 7, and 11. To find their sum, we can simply add these numbers together.

Let's calculate the sum using Python code to ensure accuracy:

```python
# Calculate the sum of the first 5 prime numbers
prime_numbers = [2, 3, 5, 7, 11]
sum_of_primes = sum(prime_numbers)
sum_of_primes
```

```output
28
```

Thus, the sum of the first 5 prime numbers is \\(\\boxed{28}\\)."""

    code_begin = "```python\n"
    code_end = "```\n"
    
    print("Debugging output ending...")
    print(f"Output ends with: {repr(output[-20:])}")
    print(f"Does output end with code_end? {output.endswith(code_end)}")
    print(f"Code_end: {repr(code_end)}")
    
    # Find all occurrences of code_end
    positions = []
    start = 0
    while True:
        pos = output.find(code_end, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    
    print(f"All code_end positions: {positions}")
    
    if positions:
        last_pos = positions[-1]
        print(f"Last code_end at position: {last_pos}")
        print(f"Text after last code_end: {repr(output[last_pos + len(code_end):])}")
        
        # Check if there's a code_begin after the last code_end
        last_code_begin = output.rfind(code_begin)
        print(f"Last code_begin at position: {last_code_begin}")
        
        if last_code_begin > last_pos:
            print("✅ There's a code_begin after the last code_end - should execute")
        else:
            print("❌ No code_begin after the last code_end - won't execute")

if __name__ == "__main__":
    debug_output_ending() 