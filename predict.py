from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image
import os
from shapely.geometry import shape, mapping
from rasterio import features

import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegFormerB5(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        print("  Loading pretrained SegFormer-B5 from HuggingFace...")
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        outputs = self.segformer(pixel_values=x, return_dict=True)
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        return upsampled_logits

class Predictor(BasePredictor):
    def setup(self):
        """Load model weights into memory"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = 8 # UAVid classes
        
        # Note: SegFormerB5 must be defined in this file
        try:
            self.model = SegFormerB5(num_classes=self.num_classes).to(self.device)
        except NameError:
            raise NameError("SegFormerB5 class not found. Please paste the class definition at the top of predict.py")
        
        # 2. Load weights (They are now baked into the image)
        weights_path = "best_model.pth"
        if not os.path.exists(weights_path):
            raise FileNotFoundError("Model weights not found! Check the build process.")
            
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Remove torch.compile prefix if present
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {key.replace('_orig_mod.', ''): v for key, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(
        self,
        image: Path = Input(description="Input image to segment"),
        classes: str = Input(description="Comma-separated list of classes to extract (e.g. 'building,road'). Leave empty for all.", default=""),
        threshold: float = Input(description="Confidence threshold", default=0.65)
    ) -> dict:
        """Run inference and return GeoJSON features"""
        # Load image
        img = Image.open(image).convert("RGB")
        img_np = np.array(img).astype(np.float32)
        
        # Match training preprocessing (Center crop to 1024x1024)
        h, w, _ = img_np.shape
        if h > 1024 or w > 1024:
            top = (h - 1024) // 2
            left = (w - 1024) // 2
            img_crop = img_np[top:top+1024, left:left+1024]
        else:
            img_crop = img_np
            
        # Normalize and prepare tensor
        img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()

        CLASS_NAMES = ["background", "building", "road", "static_car", "tree", "vegetation", "human", "moving_car"]
        results = []
        
        # Parse selected classes
        selected_classes = [c.strip().lower() for c in classes.split(",") if c.strip()]
        
        for class_idx in range(1, self.num_classes):
            class_name = CLASS_NAMES[class_idx]
            
            # Skip if user specifically requested other classes
            if selected_classes and class_name not in selected_classes:
                continue
                
            mask = (pred == class_idx).astype(np.uint8)
            
            # Simple polygonization
            shapes = features.shapes(mask, mask=mask, connectivity=4)
            geojson_features = []
            count = 0
            
            for geom, val in shapes:
                if val == 1:
                    geojson_features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {"class": CLASS_NAMES[class_idx]}
                    })
                    count += 1
            
            if count > 0:
                results.append({
                    "class": CLASS_NAMES[class_idx],
                    "count": count,
                    "geojson_url": None, # This will be handled by Replicate output or our post-processing if needed
                    "features": {
                        "type": "FeatureCollection",
                        "features": geojson_features
                    }
                })

        return {"results": results}
