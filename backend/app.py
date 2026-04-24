import io
import os
import sys
import base64
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGES_DIR = BASE_DIR / ".packages"
LOCAL_MODEL_DIR = BASE_DIR / "models" / "fashn-human-parser"

if LOCAL_PACKAGES_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES_DIR))

try:
    from rembg import remove
except Exception:
    remove = None

try:
    from fashn_human_parser import FashnHumanParser
except ImportError:
    FashnHumanParser = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
except Exception:
    torch = None
    SegformerForSemanticSegmentation = None
    SegformerImageProcessor = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
CLOTHING_LABEL_IDS = {
    3,  # top
    4,  # dress
    5,  # skirt
    6,  # pants
    7,  # belt
    9,  # hat
    10, # scarf
}
GARMENT_LABEL_MAP = {
    'top': {3},
    'dress': {4},
    'skirt': {5},
    'pants': {6},
    'belt': {7},
    'hat': {9},
    'scarf': {10},
}
MIN_GARMENT_PIXELS = 500

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE
CORS(app)


def create_human_parser_backend():
    if FashnHumanParser:
        try:
            return {
                'kind': 'package',
                'parser': FashnHumanParser(),
            }
        except Exception:
            pass

    if (
        np is not None
        and torch is not None
        and SegformerImageProcessor is not None
        and SegformerForSemanticSegmentation is not None
        and LOCAL_MODEL_DIR.exists()
    ):
        try:
            processor = SegformerImageProcessor.from_pretrained(
                str(LOCAL_MODEL_DIR),
                local_files_only=True,
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                str(LOCAL_MODEL_DIR),
                local_files_only=True,
            )
            model.eval()
            return {
                'kind': 'transformers',
                'processor': processor,
                'model': model,
            }
        except Exception:
            pass

    return None


human_parser = create_human_parser_backend()

def is_allowed_file(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def load_uploaded_image():
    if 'file' not in request.files:
        return None, (jsonify({'error': 'Missing image file field named "file".'}), 400)

    file = request.files['file']
    if not file or file.filename == '':
        return None, (jsonify({'error': 'No image selected.'}), 400)

    if not is_allowed_file(file.filename):
        return None, (jsonify({'error': 'Unsupported image type.'}), 400)

    try:
        image = Image.open(file.stream).convert('RGBA')
    except Exception:
        return None, (jsonify({'error': 'Uploaded file is not a valid image.'}), 400)

    return image, None

def send_png(image):
    img_io = io.BytesIO()
    image.save(img_io, format='PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

def image_to_data_url(image):
    img_io = io.BytesIO()
    image.save(img_io, format='PNG')
    encoded = base64.b64encode(img_io.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'

def is_likely_skin(r, g, b, a):
    if a == 0:
        return False

    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    return (
        r > 95
        and g > 40
        and b > 20
        and max_channel - min_channel > 15
        and abs(r - g) > 15
        and r > g
        and r > b
    )

def remove_skin_tone_pixels(image):
    """Keep only original foreground pixels while making likely skin transparent."""
    output = image.convert('RGBA')
    pixels = output.load()

    for y in range(output.height):
        for x in range(output.width):
            r, g, b, a = pixels[x, y]

            if is_likely_skin(r, g, b, a):
                pixels[x, y] = (r, g, b, 0)

    return output

def find_largest_upper_skin_box(image):
    image = image.convert('RGBA')
    width, height = image.size
    pixels = image.load()
    search_height = int(height * 0.6)
    visited = bytearray(width * search_height)
    largest = None

    for y in range(search_height):
        for x in range(width):
            index = y * width + x
            if visited[index]:
                continue

            r, g, b, a = pixels[x, y]
            if not is_likely_skin(r, g, b, a):
                visited[index] = 1
                continue

            stack = [(x, y)]
            visited[index] = 1
            area = 0
            min_x = max_x = x
            min_y = max_y = y

            while stack:
                current_x, current_y = stack.pop()
                area += 1
                min_x = min(min_x, current_x)
                max_x = max(max_x, current_x)
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y)

                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if (
                        next_x < 0
                        or next_x >= width
                        or next_y < 0
                        or next_y >= search_height
                    ):
                        continue

                    next_index = next_y * width + next_x
                    if visited[next_index]:
                        continue

                    nr, ng, nb, na = pixels[next_x, next_y]
                    if is_likely_skin(nr, ng, nb, na):
                        stack.append((next_x, next_y))

                    visited[next_index] = 1

            if area >= 50 and (largest is None or area > largest[0]):
                largest = (area, min_x, min_y, max_x, max_y)

    if largest is None:
        return None

    return largest[1:]

def is_likely_hair_or_shadow(r, g, b, a):
    if a == 0:
        return False

    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    is_dark = max_channel < 105
    is_dark_neutral = is_dark and max_channel - min_channel < 45
    is_brown = r > g >= b and r < 145 and g < 115 and b < 95

    return is_dark_neutral or is_brown

def remove_head_region_artifacts(original_foreground, clothing_image):
    face_box = find_largest_upper_skin_box(original_foreground)
    if face_box is None:
        return clothing_image

    output = clothing_image.convert('RGBA')
    pixels = output.load()
    width, height = output.size
    min_x, min_y, max_x, max_y = face_box
    face_width = max_x - min_x + 1
    face_height = max_y - min_y + 1

    clear_min_x = max(0, min_x - int(face_width * 0.8))
    clear_max_x = min(width - 1, max_x + int(face_width * 0.8))
    clear_min_y = max(0, min_y - int(face_height * 0.9))
    clear_max_y = min(height - 1, max_y + int(face_height * 0.2))
    center_x = (clear_min_x + clear_max_x) / 2
    center_y = (clear_min_y + clear_max_y) / 2
    radius_x = max(1, (clear_max_x - clear_min_x) / 2)
    radius_y = max(1, (clear_max_y - clear_min_y) / 2)

    for y in range(clear_min_y, clear_max_y + 1):
        for x in range(clear_min_x, clear_max_x + 1):
            r, g, b, a = pixels[x, y]
            normalized_distance = (
                ((x - center_x) / radius_x) ** 2
                + ((y - center_y) / radius_y) ** 2
            )

            if a > 0 and normalized_distance <= 1:
                pixels[x, y] = (r, g, b, 0)

    return output

def remove_faint_body_outlines(image):
    output = image.convert('RGBA')
    pixels = output.load()
    pixels_to_clear = []

    for y in range(output.height):
        for x in range(output.width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue

            max_channel = max(r, g, b)
            min_channel = min(r, g, b)
            channel_spread = max_channel - min_channel
            opaque_neighbors = 0

            for neighbor_y in range(max(0, y - 1), min(output.height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(output.width, x + 2)):
                    if neighbor_x == x and neighbor_y == y:
                        continue

                    if pixels[neighbor_x, neighbor_y][3] > 0:
                        opaque_neighbors += 1

            is_light_gray_outline = (
                70 <= max_channel <= 250
                and channel_spread <= 35
                and opaque_neighbors <= 4
            )

            if is_light_gray_outline:
                pixels_to_clear.append((x, y, r, g, b))

    for x, y, r, g, b in pixels_to_clear:
        pixels[x, y] = (r, g, b, 0)

    return output

def repair_small_transparent_holes(image):
    output = image.convert('RGBA')
    pixels = output.load()
    pixels_to_fill = []

    for y in range(1, output.height - 1):
        for x in range(1, output.width - 1):
            if pixels[x, y][3] > 0:
                continue

            opaque_neighbors = []
            for neighbor_y in range(y - 1, y + 2):
                for neighbor_x in range(x - 1, x + 2):
                    if neighbor_x == x and neighbor_y == y:
                        continue

                    nr, ng, nb, na = pixels[neighbor_x, neighbor_y]
                    if na > 0:
                        opaque_neighbors.append((nr, ng, nb, na))

            if len(opaque_neighbors) < 7:
                continue

            avg_r = sum(pixel[0] for pixel in opaque_neighbors) // len(opaque_neighbors)
            avg_g = sum(pixel[1] for pixel in opaque_neighbors) // len(opaque_neighbors)
            avg_b = sum(pixel[2] for pixel in opaque_neighbors) // len(opaque_neighbors)
            pixels_to_fill.append((x, y, avg_r, avg_g, avg_b))

    for x, y, r, g, b in pixels_to_fill:
        pixels[x, y] = (r, g, b, 255)

    return output

def remove_sparse_line_artifacts(image):
    output = image.convert('RGBA')
    pixels = output.load()
    pixels_to_clear = []

    for y in range(output.height):
        for x in range(output.width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue

            max_channel = max(r, g, b)
            min_channel = min(r, g, b)
            channel_spread = max_channel - min_channel
            low_color_signal = channel_spread <= 75 or max_channel < 140
            if not low_color_signal:
                continue

            opaque_neighbors = 0
            for neighbor_y in range(max(0, y - 2), min(output.height, y + 3)):
                for neighbor_x in range(max(0, x - 2), min(output.width, x + 3)):
                    if neighbor_x == x and neighbor_y == y:
                        continue

                    if pixels[neighbor_x, neighbor_y][3] > 0:
                        opaque_neighbors += 1

            if opaque_neighbors <= 14:
                pixels_to_clear.append((x, y, r, g, b))

    for x, y, r, g, b in pixels_to_clear:
        pixels[x, y] = (r, g, b, 0)

    return output

def remove_small_disconnected_artifacts(image):
    output = image.convert('RGBA')
    width, height = output.size
    pixels = output.load()
    visited = bytearray(width * height)
    components_to_clear = []

    for y in range(height):
        for x in range(width):
            index = y * width + x
            if visited[index] or pixels[x, y][3] == 0:
                visited[index] = 1
                continue

            stack = [(x, y)]
            visited[index] = 1
            component = []

            while stack:
                current_x, current_y = stack.pop()
                component.append((current_x, current_y))

                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                        continue

                    next_index = next_y * width + next_x
                    if visited[next_index]:
                        continue

                    visited[next_index] = 1
                    if pixels[next_x, next_y][3] > 0:
                        stack.append((next_x, next_y))

            if len(component) < 45:
                components_to_clear.extend(component)

    for x, y in components_to_clear:
        r, g, b, _ = pixels[x, y]
        pixels[x, y] = (r, g, b, 0)

    return output

def predict_segmentation_map(input_image):
    if human_parser is None:
        return None

    rgb_image = input_image.convert('RGB')
    segmentation = None

    if human_parser['kind'] == 'package':
        segmentation = human_parser['parser'].predict(rgb_image)
    elif human_parser['kind'] == 'transformers':
        processor = human_parser['processor']
        model = human_parser['model']
        inputs = processor(images=rgb_image, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=rgb_image.size[::-1],
                mode='bilinear',
                align_corners=False,
            )
            segmentation = upsampled.argmax(dim=1).squeeze().cpu().numpy()

    if segmentation is None:
        return None

    return segmentation

def build_mask_from_segmentation(segmentation, label_ids):
    clothing_mask = np.isin(segmentation, list(label_ids)).astype('uint8') * 255
    if int(clothing_mask.sum()) == 0:
        return None

    mask = Image.fromarray(clothing_mask, mode='L')
    mask = mask.filter(ImageFilter.MaxFilter(3))
    mask = mask.filter(ImageFilter.MinFilter(3))
    mask = mask.filter(ImageFilter.GaussianBlur(0.6))
    return mask

def apply_mask_to_image(input_image, mask):
    output = Image.new('RGBA', input_image.size, (0, 0, 0, 0))
    output.paste(input_image.convert('RGBA'), (0, 0), mask)
    return output

def extract_clothes_with_human_parser(input_image):
    segmentation = predict_segmentation_map(input_image)
    if segmentation is None:
        return None

    mask = build_mask_from_segmentation(segmentation, CLOTHING_LABEL_IDS)
    if mask is None:
        return None

    return apply_mask_to_image(input_image, mask)

def extract_garment_items(input_image):
    segmentation = predict_segmentation_map(input_image)
    if segmentation is None:
        return None

    items = []
    for category, label_ids in GARMENT_LABEL_MAP.items():
        pixel_count = int(np.isin(segmentation, list(label_ids)).sum())
        if pixel_count < MIN_GARMENT_PIXELS:
            continue

        mask = build_mask_from_segmentation(segmentation, label_ids)
        if mask is None:
            continue

        items.append({
            'category': category,
            'pixel_count': pixel_count,
            'image': image_to_data_url(apply_mask_to_image(input_image, mask)),
        })

    return items

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    input_image, error = load_uploaded_image()
    if error:
        return error

    if remove is None:
        return jsonify({'error': 'Background removal dependency is not available.'}), 503

    output_image = remove(input_image)
    return send_png(output_image)

@app.route('/extract-clothes', methods=['POST'])
def extract_clothes():
    input_image, error = load_uploaded_image()
    if error:
        return error

    parser_output = extract_clothes_with_human_parser(input_image)
    if parser_output is not None:
        return send_png(parser_output)

    if remove is None:
        return jsonify({'error': 'Clothing parser dependency is not available.'}), 503

    foreground = remove(input_image)
    clothing_only = remove_skin_tone_pixels(foreground)
    clothing_only = remove_head_region_artifacts(foreground, clothing_only)
    clothing_only = remove_faint_body_outlines(clothing_only)
    clothing_only = repair_small_transparent_holes(clothing_only)
    clothing_only = remove_sparse_line_artifacts(clothing_only)
    clothing_only = remove_small_disconnected_artifacts(clothing_only)
    return send_png(clothing_only)

@app.route('/extract-items', methods=['POST'])
def extract_items():
    input_image, error = load_uploaded_image()
    if error:
        return error

    items = extract_garment_items(input_image)
    if items is None:
        return jsonify({
            'error': 'Garment-level extraction requires the human parser model.',
        }), 503

    return jsonify({'items': items})

if __name__ == '__main__':
    app.run(debug=True)
