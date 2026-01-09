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
        """Run inference with sliding window for full coverage"""
        # 1. Load and prepare image
        img = Image.open(image).convert("RGB")
        img_np = np.array(img).astype(np.float32)
        h, w, _ = img_np.shape
        
        # Hyperparameters for tiling
        tile_size = 1024
        stride = 768  # 256px overlap
        
        # 2. Setup output buffers (on CPU to save GPU memory for large images)
        # We accumulate probabilities (softmax) for better blending in overlaps
        full_probs = np.zeros((self.num_classes, h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # 3. Sliding Window Inference
        print(f"  Processing image {w}x{h} with sliding window...")
        
        # Add padding to ensure we cover the edges
        pad_h = (tile_size - (h - tile_size) % stride) % stride if h > tile_size else (tile_size - h)
        pad_w = (tile_size - (w - tile_size) % stride) % stride if w > tile_size else (tile_size - w)
        
        img_padded = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        hp, wp, _ = img_padded.shape
        
        for y in range(0, hp - tile_size + 1, stride):
            for x in range(0, wp - tile_size + 1, stride):
                # Extract tile
                tile = img_padded[y:y+tile_size, x:x+tile_size]
                
                # Normalize and prepare tensor
                tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
                
                with torch.no_grad():
                    logits = self.model(tile_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                # Add to full buffers (crop if it extends beyond original image h/w)
                # But here we just iterate within padded space and then crop the final
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Calculate how much of this tile is useful (within original image)
                tile_h = y_end - y
                tile_w = x_end - x
                
                if tile_h > 0 and tile_w > 0:
                    full_probs[:, y:y_end, x:x_end] += probs[:, :tile_h, :tile_w]
                    count_map[y:y_end, x:x_end] += 1.0

        # 4. Final blending and argmax
        # Ignore areas where count_map is 0 (shouldn't happen with our padding)
        count_map = np.maximum(count_map, 1.0)
        full_probs /= count_map
        pred = full_probs.argmax(axis=0)

        # 5. Post-processing: Generate results and mask
        CLASS_NAMES = ["background", "building", "road", "static_car", "tree", "vegetation", "human", "moving_car"]
        PALETTE = [
            (0, 0, 0),       # Background
            (128, 0, 0),     # Building (Red-ish)
            (128, 64, 128),  # Road (Purple)
            (192, 0, 192),   # Static Car (Magenta)
            (0, 128, 0),     # Tree (Green)
            (128, 128, 0),   # Vegetation (Olive)
            (64, 64, 0),     # Human (Dark Yellow)
            (0, 0, 128),     # Moving Car (Blue)
        ]

        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        results = []
        selected_classes = [c.strip().lower() for c in classes.split(",") if c.strip()]
        
        for class_idx in range(0, self.num_classes):
            class_name = CLASS_NAMES[class_idx]
            color = PALETTE[class_idx]
            
            mask = (pred == class_idx).astype(np.uint8)
            
            # Add to colored mask if not background
            if class_idx > 0:
                mask_rgba[pred == class_idx] = list(color) + [160]

            # Vectorization skip logic
            if class_idx == 0: continue
            if selected_classes and class_name not in selected_classes: continue
                
            # Vectorize
            # Set a minimum area to filter small noise if needed
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
                    "features": {
                        "type": "FeatureCollection",
                        "features": geojson_features
                    }
                })

        # Save mask
        mask_path = "/tmp/mask.png"
        mask_img = Image.fromarray(mask_rgba, "RGBA")
        mask_img.save(mask_path)

        return {
            "results": results,
            "mask": Path(mask_path)
        }
