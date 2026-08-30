# Multi-Image Aspect Ratio Composer

An advanced ComfyUI node that combines several images into a single output image with a
specific aspect ratio.

## Features

### 🎛️ Dynamic input control
- **Input Count Selector**: choose between 1 and 8 input images
- **Update button**: updates the node inputs dynamically
- **Automatic UI adjustment**: the interface adapts to the chosen count on its own

### 📐 Aspect ratio presets
- **1:1 (Square)**: square format
- **4:3 (Standard)**: classic photo format
- **3:4 (Portrait)**: portrait format
- **16:9 (Widescreen)**: widescreen format
- **9:16 (Vertical)**: vertical widescreen
- **21:9 (Ultrawide)**: ultra-wide format
- **9:21 (Ultra Vertical)**: ultra-vertical format
- **3:2 (Photo)**: standard photo format
- **2:3 (Photo Portrait)**: photo portrait format
- **5:4 (Classic)**: classic format
- **4:5 (Classic Portrait)**: classic portrait format
- **16:10 (Monitor)**: monitor format
- **10:16 (Monitor Portrait)**: monitor portrait format
- **2:1 (Panorama)**: panorama format
- **1:2 (Vertical Panorama)**: vertical panorama

### 🎯 Megapixel selection
- **0.5 MP to 32 MP**: a range of resolution options
- **Automatic calculation**: width and height are derived automatically
- **Divisible by 64**: every output dimension is a multiple of 64

### 🎨 Arrangement options
- **Horizontal**: images side by side
- **Vertical**: images stacked
- **Smart Grid**: intelligent grid layout (NEW!)
  - Optimised for the chosen aspect ratio
  - Uses ALL images, so none are dropped
  - Flexible rows that may hold different numbers of images
- **Classic Grid**: traditional rigid grid layout

### ⚙️ Advanced options
- **Spacing**: gap between images (0-100 pixels)
- **Background Color**: background colour (black, white, transparent)
- **Automatic scaling**: images are scaled and centre-cropped automatically

### 🎭 Face detection (NEW!)
- **Face Detection**: on/off switch for face-aware cropping
- **Haar Cascade**: fast face detection with OpenCV
- **DNN Face**: deep-learning face detection, where available
- **Confidence**: adjustable detection strictness (1.1 - 3.0)
- **Smart cropping**: images are centred on the detected faces
- **Fallback**: falls back to a centre crop when no face is found

## Usage

### Basic steps
1. **Set the input count**: choose how many input images you want (1-8)
2. **Click Update Inputs**: rebuilds the node with that many image inputs
3. **Choose an aspect ratio**: pick the ratio you want
4. **Set the megapixels**: choose the target resolution
5. **Choose an arrangement**: horizontal, vertical, smart grid or classic grid
6. **Configure face detection**:
   - **Disabled**: standard centre cropping
   - **Haar Cascade**: fast face detection
   - **DNN Face**: advanced face detection
7. **Adjust the confidence**: detection strictness (higher = stricter)
8. **Connect the images**: wire your images into the input slots
9. **Run**: the node builds the composed image

### Outputs
- **composed_image**: the finished composition
- **width**: width of the output image
- **height**: height of the output image
- **info**: an information string describing the composition

## Technical details

### Image processing
- **Smart cropping**: images are cropped around detected faces, or centred
- **Face detection**: OpenCV-based detection for better cropping
- **Bilinear interpolation**: high-quality scaling
- **Automatic fitting**: every image is fitted into the space available to it

### Face detection details
- **Haar Cascade**:
  - Fast, CPU-friendly detection
  - Good for frontal views
  - Confidence 1.1-1.5 recommended
- **DNN Face**:
  - Advanced deep-learning detection
  - Better accuracy across varied angles
  - Somewhat slower than Haar Cascade
- **Multiple faces**:
  - With several faces the largest one is used
  - Falls back to a centre crop when none is found
- **Debug output**:
  - The console prints the faces that were detected
  - Useful when troubleshooting

### Smart Grid algorithm (NEW!)
The smart grid optimises the arrangement based on:
- **Target aspect ratio**: computes the best row/column split
- **Using every image**: no image is dropped any more
- **Flexible layouts**: rows may hold different numbers of images

**Examples for 8 images:**
- **16:9 target**: layout [4, 4] (2 rows of 4)
- **1:1 target**: layout [3, 3, 2] (3 rows: 3+3+2)
- **9:16 target**: layout [2, 2, 2, 2] (4 rows of 2)

### Classic Grid layout
- **1 image**: 1x1 grid
- **2 images**: 2x1 grid
- **3-4 images**: 2x2 grid
- **5-6 images**: 3x2 grid
- **7-8 images**: 4x2 grid (⚠️ can drop images)

### Dimension calculation
```python
# Example for 16:9 at 2 MP:
total_pixels = 2_000_000
ratio = 16/9
height = sqrt(total_pixels / ratio)
width = height * ratio
# Round to the nearest multiple of 64
width = round(width / 64) * 64
height = round(height / 64) * 64
```

## Examples

### Horizontal layout
- 3 images side by side
- 16:9 aspect ratio
- 4 MP resolution
- Result: 2560x1440 pixels

### Grid layout
- 4 images in a 2x2 arrangement
- 1:1 aspect ratio
- 8 MP resolution
- Result: 2816x2816 pixels

### Vertical layout
- 2 images stacked
- 9:16 aspect ratio
- 2 MP resolution
- Result: 1088x1920 pixels

### Portrait composition with face detection
- 4 portrait images in a 2x2 grid
- 1:1 aspect ratio
- Face detection: Haar Cascade
- Confidence: 1.3
- Result: every face nicely centred

## Tips

1. **Image quality**: images of a similar resolution give the best result
2. **Use spacing**: a gap between images separates them more clearly
3. **Grid for many images**: with 4 or more images a grid layout is usually best
4. **Adjust the megapixels**: higher for quality, lower for speed
5. **Mind the aspect ratio**: pick the ratio that suits what you are making
6. **Face detection for portraits**: turn it on when composing portraits
7. **Adjust the confidence**: lower values (1.1-1.3) detect more, higher ones (1.5-2.0) detect
   more precisely
8. **Performance**: face detection slows processing down, so turn it off when you do not need it
9. **Debugging**: the console output lists the faces that were detected

## Compatibility

- **ComfyUI**: fully compatible
- **Torch**: uses PyTorch for image processing
- **Memory**: optimised for a range of memory sizes
- **Batch processing**: supported

## Troubleshooting

### Common problems
1. **No images visible**: check that every image input you wanted is actually connected
2. **Wrong dimensions**: click "Update Inputs" after changing the input count
3. **Out of memory**: reduce the megapixel setting
4. **Loss of quality**: raise the megapixel setting, or use fewer images

### Debug information
The node prints detailed debug information to the console:
- Target dimensions
- Aspect ratio
- Number of images processed
- Final output dimensions
