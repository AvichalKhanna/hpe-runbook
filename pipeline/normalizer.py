import json
import re
import unicodedata
from pathlib import Path
from config.settings import BASE_DIR

PROTECTED_COMMANDS = {'kubectl', 'docker', 'helm', 'aws', 'git', 'python', 'pip', 'apt', 'yum', 'rpm', 'systemctl', 'curl', 'wget', 'grep', 'sed', 'awk', 'cat', 'ps', 'kill', 'export', 'chmod', 'chown'}

COMMON_MISSPELLINGS = {
    "crashlop": "crashloop", "crashlooping": "crashlooping", "exahusted": "exhausted",
    "threashold": "threshold", "connectoin": "connection", "databse": "database",
    "deploymnet": "deployment", "kubernetes": "kubernetes", "posgres": "postgres",
    "postgress": "postgres", "moniter": "monitor", "monitorring": "monitoring"
}

def load_abbreviations() -> dict:
    abbrev_path = BASE_DIR / "config" / "abbreviations.json"
    if abbrev_path.exists():
        with open(abbrev_path, 'r') as f:
            return json.load(f)
    return {}

_ABBREVIATIONS = load_abbreviations()

def normalize(query: str) -> str:
    try:
        norm_query = unicodedata.normalize('NFKC', query).strip()
        norm_query = re.sub(r'\s+', ' ', norm_query)
        
        tokens = norm_query.split()
        normalized_tokens = []
        
        for token in tokens:
            if re.match(r'^[A-Za-z]:\\\\|^/[a-z]|\\.(md|pdf|docx|yaml|yml|json|log|conf|cfg|sh|py|js)$', token):
                normalized_tokens.append(token)
                continue
            if token in PROTECTED_COMMANDS:
                normalized_tokens.append(token)
                continue
            if token.startswith('-'):
                normalized_tokens.append(token)
                continue
            if token.isupper() and len(token) > 2 and token not in ('THE', 'AND', 'FOR', 'HOW', 'WHY', 'WHAT', 'WHO', 'WHEN'):
                normalized_tokens.append(token)
                continue
            if token.startswith('"') and token.endswith('"'):
                normalized_tokens.append(token)
                continue
            
            t_lower = token.lower()
            if t_lower in _ABBREVIATIONS:
                t_lower = _ABBREVIATIONS[t_lower]
            if t_lower in COMMON_MISSPELLINGS:
                t_lower = COMMON_MISSPELLINGS[t_lower]
            
            t_lower = re.sub(r'[.,;:!?]$', '', t_lower)
            normalized_tokens.append(t_lower)
            
        return ' '.join(normalized_tokens)
    except Exception:
        return query
