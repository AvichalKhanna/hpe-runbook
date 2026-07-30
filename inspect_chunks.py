import json
with open('index_store/chunks.json') as f:
    chunks = json.load(f)
print(f'Total chunks: {len(chunks)}')
files = list(set(c["file"] for c in chunks))
print('Indexed files:')
for f in files:
    print(f'  - {f}')
