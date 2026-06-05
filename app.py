import streamlit as st
import pandas as pd
from PIL import Image
from src.utils import convert_to_url, download_image
from pipeline import build_pipeline, generate_explanation

st.set_page_config(page_title="Shop-The-Look Discovery", layout="wide")
st.title("🛍️ Shop-the-Look: Visual Product Discovery")

# Cache resource ensures model downloading and catalog indexing only happens ONCE
@st.cache_resource
def load_system():
    return build_pipeline()

# Initialize the backend pipeline
if 'pipeline_loaded' not in st.session_state:
    with st.spinner("Initializing Models & Indexing Catalog..."):
        try:
            st.session_state.detector, st.session_state.embedder, st.session_state.indexer = load_system()
            st.session_state.pipeline_loaded = True
            st.success("System Engine Ready!")
        except Exception as e:
            st.error(f"Error loading system pipeline: {e}")
            st.stop()

# Layout splits: Uploading section on the left, results on the right
col_upload, col_results = st.columns([1, 1.2], gap="large")

with col_upload:
    st.subheader("Step 1: Upload Inspiration")
    uploaded_file = st.file_uploader(
        "Drag and drop or browse a snapshot image (Dress, Shoes, Sunglasses, etc.)", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        # Open and display the user's uploaded image
        scene_img = Image.open(uploaded_file).convert("RGB")
        st.image(scene_img, caption="Your Uploaded Look", use_container_width=True)
        discover_clicked = st.button("Find Similar Products", type="primary")
    else:
        discover_clicked = False

with col_results:
    st.subheader("Step 2: Discovered Matches")
    
    if uploaded_file is None:
        st.info("Upload an inspiration image on the left to discover matching catalog items.")
        
    if discover_clicked and uploaded_file is not None:
        detector = st.session_state.detector
        embedder = st.session_state.embedder
        indexer = st.session_state.indexer
        
        with st.spinner("Analyzing style elements and querying catalog..."):
            # 1. Run YOLO Object Detection to isolate items (e.g., dress, bags, shoes)
            crops = detector.detect_and_crop(scene_img)
            
            for i, crop in enumerate(crops):
                st.write(f"---")
                st.write(f"📂 **Detected Fashion Element #{i+1}**")
                
                # Show what sub-item the system detected
                st.image(crop, width=150, caption="Isolated item crop")
                
                # 2. Extract visual vector features
                crop_emb = embedder.get_image_embedding(crop)
                
                # 3. Query FAISS index for top 2 matches
                matches = indexer.search(crop_emb, top_k=2)
                
                if not matches:
                    st.info("No close catalog matches found in the active dataset sample.")
                    continue
                
                # Render matches horizontally
                m_cols = st.columns(len(matches))
                for idx, match in enumerate(matches):
                    with m_cols[idx]:
                        p_url = convert_to_url(match['product_id'])
                        p_img = download_image(p_url)
                        
                        if p_img:
                            st.image(p_img, caption=f"Match Confidence: {match['score']*100:.1f}%", use_container_width=True)
                        else:
                            st.warning(f"Catalog Product: {match['product_id']} (Image Link Broken)")
                        
                        st.caption(f"**Product ID:** `{match['product_id']}`")
                        st.info(generate_explanation(match['score']))