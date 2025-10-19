#!/usr/bin/env python3
import sys

with open('orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    for j, char in enumerate(line):
        if ord(char) > 127:
            print(f"Line {i}, col {j}: {repr(char)} (U+{ord(char):04X})")
            print(f"  Context: {line[max(0,j-20):j+20]}")
