import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math

# Try to import OpenCV for face detection
try:
    import cv2
    OPENCV_AVAILABLE = True
    print("✓ OpenCV available for face detection")
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠ OpenCV not available - face detection disabled")


class MultiImageAspectRatioComposer:
    """
    Multi-Image Aspect Ratio Composer Node
    
    Combines multiple input images into a single output image with a specified aspect ratio.
    Features:
    - Dynamic input count selector (1-8 images)
    - Various aspect ratio presets
    - Megapixel target resolution
    - Automatic scaling and center cropping
    - Output dimensions always divisible by 64
    """
    
    # Aspect ratio presets
    ASPECT_RATIOS = {
        "1:1 (Square)": (1, 1),
        "4:3 (Standard)": (4, 3),
        "3:4 (Portrait)": (3, 4),
        "16:9 (Widescreen)": (16, 9),
        "9:16 (Vertical)": (9, 16),
        "21:9 (Ultrawide)": (21, 9),
        "9:21 (Ultra Vertical)": (9, 21),
        "3:2 (Photo)": (3, 2),
        "2:3 (Photo Portrait)": (2, 3),
        "5:4 (Classic)": (5, 4),
        "4:5 (Classic Portrait)": (4, 5),
        "16:10 (Monitor)": (16, 10),
        "10:16 (Monitor Portrait)": (10, 16),
        "2:1 (Panorama)": (2, 1),
        "1:2 (Vertical Panorama)": (1, 2),
    }
    
    # Megapixel options
    MEGAPIXELS = {
        "0.5 MP": 0.5,
        "1 MP": 1.0,
        "2 MP": 2.0,
        "4 MP": 4.0,
        "6 MP": 6.0,
        "8 MP": 8.0,
        "12 MP": 12.0,
        "16 MP": 16.0,
        "24 MP": 24.0,
        "32 MP": 32.0,
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        face_detection_options = ["disabled"]
        if OPENCV_AVAILABLE:
            face_detection_options.extend(["haar_cascade", "dnn_face"])

        return {
            "required": {
                "input_count": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "aspect_ratio": (list(cls.ASPECT_RATIOS.keys()), {"default": "16:9 (Widescreen)"}),
                "megapixels": (list(cls.MEGAPIXELS.keys()), {"default": "2 MP"}),
                "arrangement": (["horizontal", "vertical", "smart_grid", "classic_grid"], {"default": "horizontal"}),
                "spacing": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "background_color": (["black", "white", "transparent"], {"default": "black"}),
            },
            "optional": {
                "face_detection": (face_detection_options, {"default": "disabled"}),
                "face_detection_confidence": ("FLOAT", {"default": 1.3, "min": 1.1, "max": 3.0, "step": 0.1}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-evaluation when input_count changes to update the UI
        return float("nan")

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("composed_image", "width", "height", "info")
    FUNCTION = "compose_images"
    CATEGORY = "denrakeiw/image"

    OUTPUT_NODE = False

    def calculate_target_dimensions(self, aspect_ratio_key, megapixels_key):
        """Calculate target width and height based on aspect ratio and megapixels"""
        aspect_w, aspect_h = self.ASPECT_RATIOS[aspect_ratio_key]
        target_mp = self.MEGAPIXELS[megapixels_key]
        
        # Calculate total pixels
        total_pixels = target_mp * 1_000_000
        
        # Calculate dimensions maintaining aspect ratio
        ratio = aspect_w / aspect_h
        height = math.sqrt(total_pixels / ratio)
        width = height * ratio
        
        # Round to nearest multiple of 64
        width = int(round(width / 64) * 64)
        height = int(round(height / 64) * 64)
        
        # Ensure minimum size
        width = max(width, 64)
        height = max(height, 64)
        
        return width, height

    def detect_faces(self, image_np, detection_method="haar_cascade", confidence=1.3):
        """Detect faces in image and return face centers"""
        if not OPENCV_AVAILABLE or detection_method == "disabled":
            return []

        try:
            # Convert to grayscale for face detection
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            faces = []

            if detection_method == "haar_cascade":
                # Use Haar cascade classifier
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                detected_faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=confidence,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                for (x, y, w, h) in detected_faces:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    faces.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'area': w * h
                    })

            elif detection_method == "dnn_face":
                # Use DNN face detector (more accurate but slower)
                try:
                    # Load DNN model (this is a simplified version)
                    # In practice, you'd need to download the model files
                    # For now, fallback to Haar cascade
                    return self.detect_faces(image_np, "haar_cascade", confidence)
                except:
                    return self.detect_faces(image_np, "haar_cascade", confidence)

            return faces

        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def get_face_center(self, faces, image_width, image_height):
        """Calculate the center point for cropping based on detected faces"""
        if not faces:
            return None

        if len(faces) == 1:
            # Single face - use its center
            return faces[0]['center']
        else:
            # Multiple faces - use center of largest face or average center
            # Option 1: Use largest face
            largest_face = max(faces, key=lambda f: f['area'])
            return largest_face['center']

            # Option 2: Use average center (uncomment if preferred)
            # avg_x = sum(f['center'][0] for f in faces) / len(faces)
            # avg_y = sum(f['center'][1] for f in faces) / len(faces)
            # return (int(avg_x), int(avg_y))

    def get_image_aspect_ratio(self, image_tensor):
        """Get the aspect ratio of an image"""
        _, height, width, _ = image_tensor.shape
        return width / height

    def resize_and_crop_image(self, image_tensor, target_width, target_height, face_detection="disabled", face_confidence=1.3, preserve_aspect=False):
        """Resize and crop image to target dimensions with optional face detection and aspect preservation"""
        # Convert from ComfyUI format [B, H, W, C] to [B, C, H, W]
        image = image_tensor.permute(0, 3, 1, 2)

        batch_size, channels, orig_height, orig_width = image.shape
        orig_aspect = orig_width / orig_height
        target_aspect = target_width / target_height

        # Face detection for intelligent cropping
        face_center = None
        if face_detection != "disabled" and OPENCV_AVAILABLE:
            # Convert first image in batch to numpy for face detection
            img_np = image_tensor[0].cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            faces = self.detect_faces(img_np, face_detection, face_confidence)
            if faces:
                face_center = self.get_face_center(faces, orig_width, orig_height)
                print(f"✓ Detected {len(faces)} face(s), cropping around center: {face_center}")
            else:
                print("⚠ No faces detected, using center crop")

        # Smart scaling strategy
        if preserve_aspect and abs(orig_aspect - target_aspect) > 0.3:
            # For very different aspect ratios, use fit instead of fill
            scale_w = target_width / orig_width
            scale_h = target_height / orig_height
            scale = min(scale_w, scale_h)  # Scale to fit (may leave borders)

            # Calculate new dimensions
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)

            # Resize image
            resized = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)

            # Create background and center the resized image
            device = image.device
            background = torch.zeros((batch_size, channels, target_height, target_width), device=device)

            start_y = (target_height - new_height) // 2
            start_x = (target_width - new_width) // 2

            background[:, :, start_y:start_y + new_height, start_x:start_x + new_width] = resized
            result = background
        else:
            # Standard crop-to-fill approach
            scale_w = target_width / orig_width
            scale_h = target_height / orig_height
            scale = max(scale_w, scale_h)  # Scale to cover the target area

            # Calculate new dimensions after scaling
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)

            # Resize image
            resized = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)

            # Calculate crop position
            if face_center and face_detection != "disabled":
                # Scale face center coordinates
                scaled_face_x = int(face_center[0] * scale)
                scaled_face_y = int(face_center[1] * scale)

                # Calculate crop position centered on face
                start_x = max(0, min(scaled_face_x - target_width // 2, new_width - target_width))
                start_y = max(0, min(scaled_face_y - target_height // 2, new_height - target_height))
            else:
                # Default center crop
                start_y = (new_height - target_height) // 2
                start_x = (new_width - target_width) // 2

            result = resized[:, :, start_y:start_y + target_height, start_x:start_x + target_width]

        # Convert back to ComfyUI format [B, H, W, C]
        return result.permute(0, 2, 3, 1)

    def create_background(self, width, height, color, device):
        """Create background image"""
        if color == "black":
            bg = torch.zeros((1, height, width, 3), device=device)
        elif color == "white":
            bg = torch.ones((1, height, width, 3), device=device)
        else:  # transparent
            bg = torch.zeros((1, height, width, 4), device=device)
        
        return bg

    def arrange_images_horizontal(self, images, target_width, target_height, spacing, face_detection="disabled", face_confidence=1.3):
        """Arrange images horizontally"""
        num_images = len(images)
        if num_images == 0:
            return None

        # Calculate individual image width
        total_spacing = spacing * (num_images - 1)
        individual_width = (target_width - total_spacing) // num_images

        # Resize all images to the same height and calculated width
        resized_images = []
        for img in images:
            resized = self.resize_and_crop_image(img, individual_width, target_height, face_detection, face_confidence)
            resized_images.append(resized)
        
        # Concatenate horizontally with spacing
        if spacing > 0:
            # Create spacing tensors
            device = resized_images[0].device
            spacing_tensor = torch.zeros((1, target_height, spacing, 3), device=device)
            
            # Interleave images with spacing
            result_parts = []
            for i, img in enumerate(resized_images):
                result_parts.append(img)
                if i < len(resized_images) - 1:  # Don't add spacing after last image
                    result_parts.append(spacing_tensor)
            
            result = torch.cat(result_parts, dim=2)
        else:
            result = torch.cat(resized_images, dim=2)
        
        return result

    def arrange_images_vertical(self, images, target_width, target_height, spacing, face_detection="disabled", face_confidence=1.3):
        """Arrange images vertically"""
        num_images = len(images)
        if num_images == 0:
            return None

        # Calculate individual image height
        total_spacing = spacing * (num_images - 1)
        individual_height = (target_height - total_spacing) // num_images

        # Resize all images to the same width and calculated height
        resized_images = []
        for img in images:
            resized = self.resize_and_crop_image(img, target_width, individual_height, face_detection, face_confidence)
            resized_images.append(resized)
        
        # Concatenate vertically with spacing
        if spacing > 0:
            # Create spacing tensors
            device = resized_images[0].device
            spacing_tensor = torch.zeros((1, spacing, target_width, 3), device=device)
            
            # Interleave images with spacing
            result_parts = []
            for i, img in enumerate(resized_images):
                result_parts.append(img)
                if i < len(resized_images) - 1:  # Don't add spacing after last image
                    result_parts.append(spacing_tensor)
            
            result = torch.cat(result_parts, dim=1)
        else:
            result = torch.cat(resized_images, dim=1)
        
        return result

    def calculate_optimal_grid_layout(self, num_images, target_ratio):
        """Calculate optimal grid layout based on target aspect ratio"""
        if num_images == 0:
            return []
        if num_images == 1:
            return [1]

        best_layout = None
        best_score = float('inf')

        # Test different numbers of rows
        for rows in range(1, min(num_images + 1, 6)):  # Limit to max 6 rows for practicality
            # Distribute images across rows
            base_images_per_row = num_images // rows
            extra_images = num_images % rows

            layout = []
            for row in range(rows):
                images_in_row = base_images_per_row + (1 if row < extra_images else 0)
                layout.append(images_in_row)

            # Calculate resulting aspect ratio
            max_cols = max(layout)
            actual_ratio = max_cols / rows

            # Score based on deviation from target ratio
            score = abs(actual_ratio - target_ratio)

            # Prefer layouts that use space more evenly
            variance_penalty = sum((col - base_images_per_row) ** 2 for col in layout) * 0.1
            total_score = score + variance_penalty

            if total_score < best_score:
                best_score = total_score
                best_layout = layout

        return best_layout

    def arrange_images_smart_grid(self, images, target_width, target_height, spacing, face_detection="disabled", face_confidence=1.3):
        """Arrange images in an optimized grid layout"""
        num_images = len(images)
        if num_images == 0:
            return None

        # Calculate target aspect ratio
        target_ratio = target_width / target_height

        # Get optimal layout
        layout = self.calculate_optimal_grid_layout(num_images, target_ratio)
        rows = len(layout)

        print(f"Smart Grid Layout for {num_images} images: {layout} (target ratio: {target_ratio:.2f})")

        # Calculate row heights
        total_v_spacing = spacing * (rows - 1)
        row_height = (target_height - total_v_spacing) // rows

        # Process each row
        row_tensors = []
        image_idx = 0

        for row_idx, images_in_row in enumerate(layout):
            if image_idx >= len(images):
                break

            # Get images for this row
            row_images = images[image_idx:image_idx + images_in_row]
            image_idx += images_in_row

            # Calculate width for each image in this row
            total_h_spacing = spacing * (images_in_row - 1)
            available_width = target_width - total_h_spacing
            image_width = available_width // images_in_row

            # Resize images for this row
            resized_row_images = []
            for img in row_images:
                resized = self.resize_and_crop_image(img, image_width, row_height, face_detection, face_confidence)
                resized_row_images.append(resized)

            # Add spacing between images in row
            if spacing > 0 and len(resized_row_images) > 1:
                device = resized_row_images[0].device
                spacing_tensor = torch.zeros((1, row_height, spacing, 3), device=device)

                row_parts = []
                for i, img in enumerate(resized_row_images):
                    row_parts.append(img)
                    if i < len(resized_row_images) - 1:
                        row_parts.append(spacing_tensor)

                row_tensor = torch.cat(row_parts, dim=2)
            else:
                row_tensor = torch.cat(resized_row_images, dim=2)

            # Ensure row has exact target width
            current_width = row_tensor.shape[2]
            if current_width != target_width:
                device = row_tensor.device
                if current_width < target_width:
                    # Pad to target width
                    padding_width = target_width - current_width
                    padding = torch.zeros((1, row_height, padding_width, 3), device=device)
                    row_tensor = torch.cat([row_tensor, padding], dim=2)
                elif current_width > target_width:
                    # Crop to target width
                    row_tensor = row_tensor[:, :, :target_width, :]

            row_tensors.append(row_tensor)

        # Combine rows with vertical spacing
        if spacing > 0 and len(row_tensors) > 1:
            device = row_tensors[0].device
            spacing_tensor = torch.zeros((1, spacing, target_width, 3), device=device)

            final_parts = []
            for i, row in enumerate(row_tensors):
                final_parts.append(row)
                if i < len(row_tensors) - 1:
                    final_parts.append(spacing_tensor)

            result = torch.cat(final_parts, dim=1)
        else:
            result = torch.cat(row_tensors, dim=1)

        return result

    def arrange_images_classic_grid(self, images, target_width, target_height, spacing, face_detection="disabled", face_confidence=1.3):
        """Arrange images in a classic fixed grid (original algorithm)"""
        num_images = len(images)
        if num_images == 0:
            return None

        # Calculate grid dimensions (original logic)
        if num_images == 1:
            grid_cols, grid_rows = 1, 1
        elif num_images == 2:
            grid_cols, grid_rows = 2, 1
        elif num_images <= 4:
            grid_cols, grid_rows = 2, 2
        elif num_images <= 6:
            grid_cols, grid_rows = 3, 2
        else:  # 7-8 images
            grid_cols, grid_rows = 4, 2

        print(f"Classic Grid Layout: {grid_cols}x{grid_rows} for {num_images} images")

        # Calculate individual cell dimensions
        total_h_spacing = spacing * (grid_cols - 1)
        total_v_spacing = spacing * (grid_rows - 1)
        cell_width = (target_width - total_h_spacing) // grid_cols
        cell_height = (target_height - total_v_spacing) // grid_rows

        # Resize images to cell size
        resized_images = []
        for img in images:
            resized = self.resize_and_crop_image(img, cell_width, cell_height, face_detection, face_confidence)
            resized_images.append(resized)

        # Fill empty cells with background if needed
        device = resized_images[0].device
        while len(resized_images) < grid_cols * grid_rows:
            empty_cell = torch.zeros((1, cell_height, cell_width, 3), device=device)
            resized_images.append(empty_cell)

        # Arrange in grid
        rows = []
        for row in range(grid_rows):
            row_images = []
            for col in range(grid_cols):
                idx = row * grid_cols + col
                if idx < len(resized_images):
                    row_images.append(resized_images[idx])
                    if col < grid_cols - 1 and spacing > 0:  # Add horizontal spacing
                        spacing_tensor = torch.zeros((1, cell_height, spacing, 3), device=device)
                        row_images.append(spacing_tensor)

            if row_images:
                row_tensor = torch.cat(row_images, dim=2)
                rows.append(row_tensor)

                if row < grid_rows - 1 and spacing > 0:  # Add vertical spacing
                    spacing_tensor = torch.zeros((1, spacing, target_width, 3), device=device)
                    rows.append(spacing_tensor)

        result = torch.cat(rows, dim=1)
        return result

    def arrange_images_grid(self, images, target_width, target_height, spacing, face_detection="disabled", face_confidence=1.3):
        """Arrange images in a grid - uses smart grid algorithm by default"""
        return self.arrange_images_smart_grid(images, target_width, target_height, spacing, face_detection, face_confidence)
        
        # Fill empty cells with background if needed
        device = resized_images[0].device
        while len(resized_images) < grid_cols * grid_rows:
            empty_cell = torch.zeros((1, cell_height, cell_width, 3), device=device)
            resized_images.append(empty_cell)
        
        # Arrange in grid
        rows = []
        for row in range(grid_rows):
            row_images = []
            for col in range(grid_cols):
                idx = row * grid_cols + col
                if idx < len(resized_images):
                    row_images.append(resized_images[idx])
                    if col < grid_cols - 1 and spacing > 0:  # Add horizontal spacing
                        spacing_tensor = torch.zeros((1, cell_height, spacing, 3), device=device)
                        row_images.append(spacing_tensor)
            
            if row_images:
                row_tensor = torch.cat(row_images, dim=2)
                rows.append(row_tensor)
                
                if row < grid_rows - 1 and spacing > 0:  # Add vertical spacing
                    spacing_tensor = torch.zeros((1, spacing, target_width, 3), device=device)
                    rows.append(spacing_tensor)
        
        result = torch.cat(rows, dim=1)
        return result

    def compose_images(self, input_count, aspect_ratio, megapixels, arrangement, spacing, background_color, face_detection="disabled", face_detection_confidence=1.3, **kwargs):
        """Main composition function"""
        # Calculate target dimensions
        target_width, target_height = self.calculate_target_dimensions(aspect_ratio, megapixels)
        
        print(f"=== Multi-Image Aspect Ratio Composer ===")
        print(f"Target dimensions: {target_width}x{target_height}")
        print(f"Aspect ratio: {aspect_ratio}")
        print(f"Megapixels: {megapixels}")
        print(f"Arrangement: {arrangement}")
        print(f"Input count: {input_count}")
        print(f"Face detection: {face_detection}")
        if face_detection != "disabled":
            print(f"Face detection confidence: {face_detection_confidence}")
        
        # Collect input images
        images = []
        for i in range(1, input_count + 1):
            image_key = f"image_{i}"
            if image_key in kwargs and kwargs[image_key] is not None:
                images.append(kwargs[image_key])
        
        if not images:
            # Create empty image if no inputs
            device = torch.device("cpu")
            result = self.create_background(target_width, target_height, background_color, device)
        else:
            # Arrange images based on selected arrangement
            if arrangement == "horizontal":
                result = self.arrange_images_horizontal(images, target_width, target_height, spacing, face_detection, face_detection_confidence)
            elif arrangement == "vertical":
                result = self.arrange_images_vertical(images, target_width, target_height, spacing, face_detection, face_detection_confidence)
            elif arrangement == "smart_grid":
                result = self.arrange_images_smart_grid(images, target_width, target_height, spacing, face_detection, face_detection_confidence)
            elif arrangement == "classic_grid":
                result = self.arrange_images_classic_grid(images, target_width, target_height, spacing, face_detection, face_detection_confidence)
            else:  # fallback to smart_grid
                result = self.arrange_images_smart_grid(images, target_width, target_height, spacing, face_detection, face_detection_confidence)
        
        print(f"Output shape: {result.shape}")
        print(f"Output dimensions: {result.shape[2]}x{result.shape[1]}")

        # Create info string
        actual_mp = (target_width * target_height) / 1_000_000
        face_info = f" | Face detection: {face_detection}" if face_detection != "disabled" else ""
        info = f"Composed {len(images)} images | {target_width}x{target_height} | {actual_mp:.1f}MP | {arrangement}{face_info}"

        return (result, target_width, target_height, info)
