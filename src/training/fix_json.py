r"""
Fix invalid unicode escapes in the merged JSON file.
The file has backslash-u sequences where the u is not followed by valid hex digits,
which causes json.load to fail.
"""
import re
import json
import sys

path = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop/data/items_shuffle_full.json"
out = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop/data/items_shuffle_fixed.json"

print("Reading file...")
with open(path, "rb") as f:
    raw = f.read()
print(f"Loaded {len(raw)} bytes")

# In raw bytes, \u is 0x5c 0x75
# Find all \u not followed by 4 hex digits
# Replace \u with \\u (double backslash + u)
# Only match \u that is NOT preceded by another \
# (i.e. NOT part of \\u which is a valid JSON escape for literal \u)
bad_pattern = re.compile(rb'(?<!\\)\\u(?![0-9a-fA-F]{4})')
matches = list(bad_pattern.finditer(raw))
print(f"Found {len(matches)} bad unicode escapes")
for m in matches[:5]:
    pos = m.start()
    context = raw[max(0,pos-20):pos+20]
    print(f"  Position {pos}: {context}")

# Fix: replace \u with \\u at each match position
# Work backwards to not shift positions
fixed = bytearray(raw)
offset = 0
for m in matches:
    pos = m.start() + offset
    # Replace the single \ with \\
    fixed[pos:pos+1] = b'\\\\'
    offset += 1  # we added 1 byte

print(f"Fixed. Original {len(raw)} bytes -> {len(fixed)} bytes")

with open(out, "wb") as f:
    f.write(fixed)
print(f"Written to {out}")

# Verify
print("Verifying...")
with open(out) as f:
    data = json.load(f)
    print(f"OK! Items: {len(data)}")
