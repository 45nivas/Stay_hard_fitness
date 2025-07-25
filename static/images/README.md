# Background Images for Stay Hard Fitness

## How to add background images:

1. **Local Images**: Place your images in this directory (`static/images/`)
   - `gym-background.jpg` - Main background image
   - `gym-classic.jpg` - Alternative classic gym theme
   - `fitness-motivation.jpg` - Motivational fitness background

2. **External Images**: Use the `.bg-external-image` class or modify the CSS directly

## Current Setup:

The CSS is configured to use:
- **Primary**: `/static/images/gym-background.jpg`
- **External**: Pinterest image URL (if using `.bg-external-image` class)

## Supported formats:
- JPG/JPEG
- PNG
- WebP (recommended for web)
- GIF

## Recommended image specs:
- **Resolution**: 1920x1080 or higher
- **Aspect ratio**: 16:9 or 4:3
- **File size**: Under 2MB for fast loading

## To change background:
1. Add your image file to this directory
2. Update the CSS in `static/style.css` line ~188
3. Or apply CSS classes like `.bg-external-image` to body element
