#!/usr/bin/env python3

with open('document_handler.py', 'r') as f:
    lines = f.readlines()

# Find and fix the broken lines
for i, line in enumerate(lines):
    if "sanitized += '" in line and line.strip().endswith("'"):
        continue  # This line is fine
    elif "sanitized += '" in line and not line.strip().endswith("'"):
        # This is a broken multiline string - find the end
        j = i + 1
        while j < len(lines) and not lines[j].strip().endswith("'"):
            j += 1
        if j < len(lines):
            # Combine the lines
            combined = line.rstrip() + '\\n```'  # Close unclosed code block\n'
            lines[i] = combined
            # Remove the intermediate lines
            for k in range(i + 1, j + 1):
                lines[k] = ''

    # Fix other common broken patterns
    if 'sanitized = re.sub(r\'' in line and '{\n{4,}\', \'\n\n\n\', sanitized)' in line:
        lines[i] = "        sanitized = re.sub(r'\\n{4,}', '\\n\\n\\n', sanitized)\n"

    if 'all_content.append(f"[Page {page[\'page\']}]")' in line:
        lines[i] = line  # This is probably fine

# Write the fixed content
with open('document_handler.py', 'w') as f:
    f.writelines([line for line in lines if line.strip()])