import pandas as pd
import json

def json_loadf(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)