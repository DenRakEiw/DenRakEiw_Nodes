# Denrakeiw Nodes

A custom node pack for ComfyUI that provides utility nodes for image generation and manipulation.

## Nodes

### Color Generator Node

![Screenshot 2025-07-25 175434](https://github.com/user-attachments/assets/ef53eb14-8e10-40df-9909-65825a425017)


Generates solid color images based on predefined colors.

**Features:**
- Width and height input fields (1-8192 pixels)
- Dropdown menu with 130+ predefined colors
- Outputs:
  - Solid color image
  - Color name as string
  - Hex code as string

**Available Color Categories:**
- **Basic Colors**: Red, Green, Blue, Yellow, Cyan, Magenta, White, Black
- **Grays**: Gray, Light Gray, Dark Gray, Dim Gray, Silver, Gainsboro, WhiteSmoke
- **Reds**: Dark Red, Crimson, Fire Brick, Indian Red, Light Coral, Salmon, Coral, Tomato, Orange Red, Maroon
- **Oranges**: Orange, Dark Orange, Peach, Papaya Whip, Moccasin, Peach Puff, Khaki
- **Yellows**: Light Yellow, Lemon Chiffon, Cornsilk, Gold, Goldenrod
- **Greens**: Lime, Lime Green, Lawn Green, Chartreuse, Spring Green, Forest Green, Sea Green, Olive, Yellow Green
- **Cyans/Teals**: Aqua, Dark Cyan, Teal, Turquoise, Aqua Marine, Cadet Blue
- **Blues**: Light Blue, Powder Blue, Steel Blue, Cornflower Blue, Deep Sky Blue, Royal Blue, Navy, Midnight Blue
- **Purples/Violets**: Purple, Indigo, Slate Blue, Blue Violet, Dark Orchid, Orchid, Violet, Plum, Lavender
- **Pinks**: Pink, Light Pink, Hot Pink, Deep Pink
- **Browns**: Brown, Saddle Brown, Sienna, Chocolate, Peru, Sandy Brown, Tan, Wheat
- **Special Colors**: Beige, Floral White, Ghost White, Honeydew, Ivory, Azure, Snow, Mint Cream

## Installation

1. Clone or download this repository to your ComfyUI custom_nodes directory:
   ```
   ComfyUI/custom_nodes/denrakeiw_nodes/
   ```

2. Restart ComfyUI

3. The nodes will appear under the "denrakeiw/image" category

## Requirements

- ComfyUI
- torch
- numpy
- PIL (Pillow)

## Usage

1. Add the "Color Generator" node to your workflow
2. Set the desired width and height
3. Select a color from the dropdown
4. Connect the outputs to other nodes as needed

The node will generate a solid color image and provide both the color name and hex code as strings for further processing.

