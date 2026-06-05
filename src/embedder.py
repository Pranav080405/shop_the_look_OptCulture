import torch
from fashion_clip.fashion_clip import FashionCLIP

class FashionEmbedder:
    def __init__(self):
        # Automatically downloads and instantiates FashionCLIP
        self.fclip = FashionCLIP('fashion-clip')
        
    def get_image_embedding(self, pil_image):
        """Generates a normalized embedding vector for a PIL Image."""
        # FashionCLIP natively works well with lists of images
        embeddings = self.fclip.encode_images([pil_image], batch_size=1)
        # Normalize the vector for cosine similarity matching
        embedding = embeddings[0] / (torch.linalg.norm(torch.tensor(embeddings[0])).item() + 1e-8)
        return embedding