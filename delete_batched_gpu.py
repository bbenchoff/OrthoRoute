#!/usr/bin/env python3
"""Delete _route_all_batched_gpu function from unified_pathfinder.py"""

filepath = r"C:\Users\Benchoff\Documents\GitHub\OrthoRoute\orthoroute\algorithms\manhattan\unified_pathfinder.py"

# Read file
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Original: {len(lines)} lines")

# Delete lines 2964-3596 (0-indexed: 2963-3595)
# Line 2964 is "    def _route_all_batched_gpu..."
# Line 3596 is the last line of the function (just before next function)
new_lines = lines[:2963] + lines[3596:]

print(f"After deletion: {len(new_lines)} lines (removed {len(lines) - len(new_lines)} lines)")

# Write back
with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Done!")
