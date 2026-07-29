import re
from typing import List

def compress_chunk(text: str) -> str:
    lines = text.split('\n')
    compressed = []
    
    in_code = False
    
    for line in lines:
        line_s = line.strip()
        
        if line_s.startswith('```') or line_s.startswith('~~~'):
            in_code = not in_code
            compressed.append(line)
            continue
            
        if in_code:
            compressed.append(line)
            continue
            
        prefixes = ('$', '>', '#!', 'kubectl', 'docker', 'helm', 'aws', 'git', 'python', 'pip', 'apt', 'yum', 'rpm', 'systemctl', 'curl', 'wget', 'export')
        if any(line_s.startswith(p) for p in prefixes):
            compressed.append(line)
            continue
            
        if '|' in line_s:
            compressed.append(line)
            continue
            
        if line_s.startswith('#'):
            compressed.append(line)
            continue
            
        if re.match(r'^(WARNING|CAUTION|NOTE|IMPORTANT|DANGER|\*\*WARNING|\*\*CAUTION)', line_s):
            compressed.append(line)
            continue
            
        if re.search(r'\b[A-Z_]{5,}\b', line_s):
            compressed.append(line)
            continue
            
        if '/' in line_s or '\\' in line_s:
            compressed.append(line)
            continue
            
        # Repetitive fillers
        if re.search(r'This section (describes|covers|explains|provides)', line_s, re.I):
            continue
        if re.search(r'The following (steps|procedure|commands) (describe|explain|show)', line_s, re.I):
            continue
        if re.search(r'Please (note|be aware|ensure) that', line_s, re.I):
            continue
            
        if not line_s and (not compressed or not compressed[-1].strip()):
            continue
            
        compressed.append(line)
        
    return '\n'.join(compressed).strip()

def compress_chunks(chunks: List[dict]) -> List[dict]:
    compressed_chunks = []
    for c in chunks:
        c_new = c.copy()
        c_new['text'] = compress_chunk(c['text'])
        compressed_chunks.append(c_new)
    return compressed_chunks
