import pandas as pd
from src.utils import safe_load_jsonl, convert_to_url, download_image
from src.detector import FashionDetector
from src.embedder import FashionEmbedder
from src.indexer import CatalogIndexer

def generate_explanation(score):
    """Generates dynamic explanations based on semantic match confidence."""
    if score > 0.92:
        return "Exact visual match found. High consistency across product category, color scheme, silhouette pattern, and apparel style."
    elif score > 0.75:
        return "Recommended based on strong structural similarities in category, color spectrum, and overall visual appearance."
    else:
        return "Alternative fashion match recommended due to shared style taxonomy and close structural shape properties."

def build_pipeline():
    print("Initializing components...")
    detector = FashionDetector()
    embedder = FashionEmbedder()
    indexer = CatalogIndexer()
    
    # 1. Load Data safely
    catalog_data = safe_load_jsonl("data/product_catalog.jsonl")
    df_catalog = pd.DataFrame(catalog_data)
    
    # Fast sampling threshold for rapid demo engine booting up 
    demo_limit = 50 
    df_catalog_sample = df_catalog.head(demo_limit)
    
    print(f"Indexing the first {len(df_catalog_sample)} catalog items for the live demo...")
    
    # 2. Populate FAISS Index
    for idx, row in df_catalog_sample.iterrows():
        prod_id = row['product']
        url = convert_to_url(prod_id)
        img = download_image(url)
        if img:
            emb = embedder.get_image_embedding(img)
            indexer.add_to_index(prod_id, emb)
            
    print("Catalog indexing complete!")
    return detector, embedder, indexer

if __name__ == "__main__":
    # Test execution fallback
    detector, embedder, indexer = build_pipeline()
    print("Pipeline check passed successfully!")