import random

# Read lines from the original file
with open('old_test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Shuffle lines randomly
random.shuffle(lines)

# Split the data into two halves
half = len(lines) // 2
valid_lines = lines[:half]
test_lines = lines[half:]

# Write the first half to valid.txt
with open('valid.txt', 'w', encoding='utf-8') as f:
    f.writelines(valid_lines)

# Write the second half to test.txt
with open('test.txt', 'w', encoding='utf-8') as f:
    f.writelines(test_lines)
