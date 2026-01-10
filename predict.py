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
        print(f"  Processing image {w}x{h} with sliding window (tile={tile_size}, stride={stride})...")
        
        # Calculate padding to ensure complete coverage
        # We need enough padding so that the last tile covers the bottom-right corner
        # For a tile at position y, it covers y to y+tile_size
        # The loop goes: 0, stride, 2*stride, ... up to (last_y such that last_y + tile_size <= hp)
        # We need: last_y + tile_size >= h, where last_y = n * stride for some n
        # So we need: n*stride + tile_size >= h, meaning n >= (h - tile_size) / stride
        # To ensure coverage: hp should be at least h if h <= tile_size, else large enough for loop
        
        if h <= tile_size:
            pad_h = tile_size - h  # Make it exactly tile_size
        else:
            # Ensure at least one tile covers the bottom edge
            num_tiles_h = (h - tile_size + stride - 1) // stride + 1
            required_h = (num_tiles_h - 1) * stride + tile_size
            pad_h = max(0, required_h - h)
        
        if w <= tile_size:
            pad_w = tile_size - w
        else:
            num_tiles_w = (w - tile_size + stride - 1) // stride + 1
            required_w = (num_tiles_w - 1) * stride + tile_size
            pad_w = max(0, required_w - w)
        
        img_padded = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        hp, wp, _ = img_padded.shape
        print(f"  Padded to {wp}x{hp}, will process {len(range(0, hp - tile_size + 1, stride)) * len(range(0, wp - tile_size + 1, stride))} tiles")
        
        for y in range(0, hp - tile_size + 1, stride):
            for x in range(0, wp - tile_size + 1, stride):
                # Extract tile
                tile = img_padded[y:y+tile_size, x:x+tile_size]
                
                # Normalize and prepare tensor
                tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
                
                with torch.no_grad():
                    logits = self.model(tile_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                # The tile is valid - add it to full buffers
                # Calculate the actual region to update (clipped to original image size)
                y_start_out = y
                x_start_out = x
                y_end_out = min(y + tile_size, h)
                x_end_out = min(x + tile_size, w)
                
                # Calculate corresponding region in the tile
                tile_y_end = y_end_out - y
                tile_x_end = x_end_out - x
                
                if y_start_out < h and x_start_out < w:
                    full_probs[:, y_start_out:y_end_out, x_start_out:x_end_out] += probs[:, :tile_y_end, :tile_x_end]
                    count_map[y_start_out:y_end_out, x_start_out:x_end_out] += 1.0

        # 4. Final blending and argmax
        count_map = np.maximum(count_map, 1.0)
        full_probs /= count_map
        
        # Get max probability and corresponding class
        confidences = full_probs.max(axis=0)
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
            
            # Pixels for this class must also meet the confidence threshold
            is_class = (pred == class_idx)
            is_confident = (confidences >= threshold)
            mask = (is_class & is_confident).astype(np.uint8)
            
            # Skip if no pixels found
            if not np.any(mask):
                continue

            # --- POST-PROCESSING CLEANUP ---
            # Apply morphological operations to reduce noise and fill holes
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # MORPH_CLOSE: Fill small holes inside objects
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # MORPH_OPEN: Remove small noise/artifacts outside objects  
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Median blur: Smooth jagged edges
            mask = cv2.medianBlur(mask, 3)
            
            # Re-check if any mask remains after cleanup
            if not np.any(mask):
                continue

            # Update colored mask ONLY for selected classes (if any are specified)
            should_include = not selected_classes or class_name in selected_classes
            
            if class_idx > 0 and should_include:
                mask_rgba[mask == 1] = list(color) + [160]

            # Vectorization logic: Skip background and unselected classes
            if class_idx == 0: continue
            if not should_include: continue
            
            # Save individual class mask
            class_mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            class_mask_rgba[mask == 1] = list(color) + [200]
            class_mask_path = f"/tmp/mask_{class_name}.png"
            Image.fromarray(class_mask_rgba, "RGBA").save(class_mask_path)
                
            # Vectorize
            shapes = features.shapes(mask, mask=mask, connectivity=4)
            geojson_features = []
            count = 0
            
            for geom, val in shapes:
                if val == 1:
                    geojson_features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {"class": CLASS_NAMES[class_idx], "confidence": float(np.mean(confidences[mask==1]))}
                    })
                    count += 1
            
            if count > 0:
                results.append({
                    "class": CLASS_NAMES[class_idx],
                    "count": count,
                    "features": {
                        "type": "FeatureCollection",
                        "features": geojson_features
                    },
                    "png_url": Path(class_mask_path)  # Replicate will upload this
                })

        # Save combined mask
        mask_path = "/tmp/mask.png"
        mask_img = Image.fromarray(mask_rgba, "RGBA")
        mask_img.save(mask_path)

        return {
            "results": results,
            "mask": Path(mask_path)
        }
