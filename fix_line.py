#!/usr/bin/env python3

with open('document_handler.py', 'r') as f:
    lines = f.readlines()

# Find and fix line 273
for i, line in enumerate(lines):
    if 'sanitized += "' in line and not line.rstrip().endswith('"'):
        # This is the broken line
        lines[i] = '            sanitized += "\\n```"  # Close unclosed code block\n'
        break

with open('document_handler.py', 'w') as f:
    f.writelines(lines)