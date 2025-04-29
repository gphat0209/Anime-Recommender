import pandas as pd
import numpy as np
import json, os
from sklearn.utils import shuffle
import math

df = pd.read_csv("data/preprocessed_anime_data.csv")

short_df = df[['Sypnosis', 'Genres', 'Label']]
shuffle_df = shuffle(short_df, random_state=42).reset_index(drop=True)

train_length = math.floor(0.6*len(shuffle_df))
valid_length = math.floor(0.9*len(shuffle_df))

train_df = shuffle_df.iloc[:train_length]
valid_df = shuffle_df.iloc[train_length:valid_length]
test_df = shuffle_df.iloc[valid_length:]

data_dir = 'training_data/'
os.makedirs(data_dir, exist_ok=True)

def get_prompt(df, path):
    data = []
    for _,row in df.iterrows():
        prompt = f"""You are an intelligent anime recommender system. Your task is to evaluate how good an anime is based solely on its synopsis and genres.

<Anime Details>
Synopsis: "{row['Sypnosis']}"
Genres: {row['Genres']}

Based on this information, classify the anime strictly as one of the following: "Top-tier", "Worthwhile", "Watchable", or "Terrible"."""
        
        data.append({
            'input': prompt,
            'output': row['Label']
        })
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

get_prompt(train_df, data_dir+'train.json')
get_prompt(valid_df, data_dir+'valid.json')
get_prompt(test_df, data_dir+'test.json')