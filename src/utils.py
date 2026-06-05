import os
import json
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

def convert_to_url(signature):
    """Converts a image signature hash to its Pinterest URL."""
    if not signature or pd.isna(signature):
        return None
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)

def safe_load_jsonl(file_path):
    """Safely loads JSONL lines handling trailing spaces or empty lines."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def download_image(url):
    """Downloads an image and returns a PIL Image object with error handling."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None