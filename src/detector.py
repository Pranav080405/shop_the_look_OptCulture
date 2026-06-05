import torch
from ultralytics import YOLO

class FashionDetector:
    def __init__(self):
        # Using a pre-trained general object detector or fashion model
        # 'yolov8n.pt' detects 'handbag', 'tie', 'suitcase', 'clothing' features natively
        self.model = YOLO('yolov8n.pt') 

    def detect_and_crop(self, pil_image):
        """
        Detects bounding boxes in a scene and returns a list of cropped PIL images.
        Falls back to the original image if no salient items are detected.
        """
        results = self.model(pil_image, verbose=False)[0]
        crops = []
        
        # Filter for fashion-related classes in COCO (e.g., person, handbag, tie)
        # Or simply crop high-confidence objects
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > 0.25:  # Confidence threshold
                xyxy = box.xyxy[0].tolist()
                crop = pil_image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
                crops.append(crop)
                
        # Fallback to full scene image if nothing distinct was localized
        if not crops:
            crops.append(pil_image)
        return crops