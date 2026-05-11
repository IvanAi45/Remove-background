import io
import os
import sys
import base64
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
from pathlib import Path
from urllib.parse import unquote

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGES_DIR = BASE_DIR / ".packages"
LOCAL_MODEL_DIR = BASE_DIR / "models" / "fashn-human-parser"
FRONTEND_DIR = BASE_DIR.parent / "frontend"
PERSON_MODELS_DIR = FRONTEND_DIR / "person-models"
FASHN_VTON_DIR = Path(os.environ.get(
    "FASHN_VTON_DIR",
    str(BASE_DIR / "models" / "fashn-vton-1.5"),
)).resolve()
FASHN_VTON_SRC_DIR = FASHN_VTON_DIR / "src"
FASHN_VTON_WEIGHTS_DIR = Path(os.environ.get(
    "FASHN_VTON_WEIGHTS_DIR",
    str(FASHN_VTON_DIR / "weights"),
)).resolve()

if LOCAL_PACKAGES_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES_DIR))

if FASHN_VTON_SRC_DIR.exists():
    sys.path.insert(0, str(FASHN_VTON_SRC_DIR))

try:
    from rembg import remove, new_session
except BaseException:
    remove = None
    new_session = None

try:
    from fashn_human_parser import FashnHumanParser
except Exception:
    FashnHumanParser = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
    from transformers import (
        CLIPModel,
        CLIPProcessor,
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )
except Exception:
    torch = None
    CLIPModel = None
    CLIPProcessor = None
    SegformerForSemanticSegmentation = None
    SegformerImageProcessor = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
PERSON_MODEL_CATEGORIES = {
    'Male': 'Male',
    'Female': 'Female',
    'Non-Binary': 'Non-Binary',
}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
MAX_INPUT_IMAGE_SIDE = 1024
TRYON_NUM_TIMESTEPS = int(os.environ.get('TRYON_NUM_TIMESTEPS', '12'))
TRYON_GUIDANCE_SCALE = float(os.environ.get('TRYON_GUIDANCE_SCALE', '1.5'))
CLOTHING_LABEL_IDS = {
    3,  # top
    4,  # dress
    5,  # skirt
    6,  # pants
    7,  # belt
    9,  # hat
    10, # scarf
    18, # shoe (optional parser label)
    19, # shoe (optional parser label)
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
SINGLE_ITEM_PIXEL_MIN = 800
MAX_MAJOR_COMPONENTS = 1
MAX_SECONDARY_COMPONENT_RATIO = 0.22
CLIP_MODEL_ID = 'openai/clip-vit-base-patch32'
SEGMENTATION_CATEGORY_LABELS = {
    'upper_body': {3, 4},
    'lower_body': {5, 6},
    'footwear': {18, 19},
}

CATEGORY_TAXONOMY = {
    'upper_body': [
        't_shirt',
        'tank_top_vest',
        'shirt_blouse',
        'polo_shirt',
        'hoodie_sweatshirt',
        'sweater_pullover',
        'cardigan',
        'suit_jacket',
        'jacket',
        'trench_coat',
        'overcoat',
        'down_jacket',
        'dress',
    ],
    'lower_body': [
        'jeans',
        'dress_pants',
        'sweatpants_joggers',
        'leggings',
        'casual_shorts',
        'sports_shorts',
        'mini_skirt',
        'maxi_skirt',
        'pleated_skirt',
    ],
    'footwear': [
        'sneakers',
        'skate_shoes',
        'running_shoes',
        'oxfords',
        'loafers',
        'derby_shoes',
        'ankle_boots',
        'high_boots',
        'martin_boots',
        'sandals',
        'slippers',
        'flip_flops',
    ],
}

CATEGORY_PROMPTS = {
    'upper_body': 'a catalog photo of one upper-body clothing item',
    'lower_body': 'a catalog photo of one lower-body clothing item',
    'footwear': 'a catalog photo of one footwear item',
}

SUBCATEGORY_PROMPTS = {
    't_shirt': 'a catalog photo of one t-shirt',
    'tank_top_vest': 'a catalog photo of one tank top or vest',
    'shirt_blouse': 'a catalog photo of one shirt or blouse',
    'polo_shirt': 'a catalog photo of one polo shirt',
    'hoodie_sweatshirt': 'a catalog photo of one hoodie or sweatshirt',
    'sweater_pullover': 'a catalog photo of one sweater or pullover',
    'cardigan': 'a catalog photo of one cardigan',
    'suit_jacket': 'a catalog photo of one suit jacket',
    'jacket': 'a catalog photo of one jacket',
    'trench_coat': 'a catalog photo of one trench coat',
    'overcoat': 'a catalog photo of one overcoat',
    'down_jacket': 'a catalog photo of one down jacket',
    'dress': 'a catalog photo of one dress',
    'jeans': 'a catalog photo of one pair of jeans',
    'dress_pants': 'a catalog photo of one pair of dress pants',
    'sweatpants_joggers': 'a catalog photo of one pair of sweatpants or joggers',
    'leggings': 'a catalog photo of one pair of leggings',
    'casual_shorts': 'a catalog photo of one pair of casual shorts',
    'sports_shorts': 'a catalog photo of one pair of sports shorts',
    'mini_skirt': 'a catalog photo of one mini skirt',
    'maxi_skirt': 'a catalog photo of one maxi skirt',
    'pleated_skirt': 'a catalog photo of one pleated skirt',
    'sneakers': 'a catalog photo of one sneaker shoe item',
    'skate_shoes': 'a catalog photo of one skate shoe item',
    'running_shoes': 'a catalog photo of one running shoe item',
    'oxfords': 'a catalog photo of one oxford shoe item',
    'loafers': 'a catalog photo of one loafer shoe item',
    'derby_shoes': 'a catalog photo of one derby shoe item',
    'ankle_boots': 'a catalog photo of one ankle boot item',
    'high_boots': 'a catalog photo of one high boot item',
    'martin_boots': 'a catalog photo of one martin boot item',
    'sandals': 'a catalog photo of one pair of sandals',
    'slippers': 'a catalog photo of one pair of slippers',
    'flip_flops': 'a catalog photo of one pair of flip flops',
}
SUBCATEGORY_TO_CATEGORY = {
    subcategory: category
    for category, subcategories in CATEGORY_TAXONOMY.items()
    for subcategory in subcategories
}
FAST_CLASSIFY_IMAGE_SIDE = 256
MAX_PREVIEW_WORKERS = max(2, min(6, (os.cpu_count() or 4)))

clip_classifier = None
clip_text_feature_cache = {}
rembg_session = None
preview_cache = {}
preview_lock = threading.Lock()
tryon_pipeline = None
tryon_lock = threading.Lock()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE
CORS(app)


@app.route('/', methods=['GET'])
def home():
    """
    Serves the frontend entry page so local testing works on the backend URL.
    """
    index_file = FRONTEND_DIR / 'index.html'
    if index_file.exists():
        return send_from_directory(str(FRONTEND_DIR), 'index.html')

    return jsonify({
        'message': 'Clothing API is running.',
        'hint': 'Open frontend/index.html or restore the frontend folder.',
    })


@app.route('/health', methods=['GET'])
def health():
    """
    Returns a lightweight status payload for connection checks.
    """
    return jsonify({
        'status': 'ok',
        'human_parser_ready': human_parser is not None,
        'remove_bg_ready': remove is not None,
    })


@app.route('/person-models', methods=['GET'])
def list_person_models():
    """
    Returns bundled try-on model photos grouped by preset category.
    """
    categories = []
    for category, display_name in PERSON_MODEL_CATEGORIES.items():
        category_dir = PERSON_MODELS_DIR / category
        models = []
        if category_dir.exists():
            for file_path in sorted(category_dir.iterdir()):
                if file_path.is_file() and is_allowed_file(file_path.name):
                    models.append({
                        'id': f'{category}/{file_path.name}',
                        'category': category,
                        'display_name': display_name,
                        'filename': file_path.name,
                        'image_url': f'/person-models/{category}/{file_path.name}',
                    })

        categories.append({
            'category': category,
            'display_name': display_name,
            'models': models,
        })

    return jsonify({
        'ok': True,
        'categories': categories,
    })


@app.route('/person-models/<category>/<filename>', methods=['GET'])
def serve_person_model(category, filename):
    """
    Serves one bundled try-on model image.
    """
    if category not in PERSON_MODEL_CATEGORIES:
        return jsonify({'error': 'Unknown model category.'}), 404

    safe_filename = Path(filename).name
    if not safe_filename or not is_allowed_file(safe_filename):
        return jsonify({'error': 'Unsupported model image.'}), 404

    category_dir = PERSON_MODELS_DIR / category
    model_path = category_dir / safe_filename
    if not model_path.exists() or not model_path.is_file():
        return jsonify({'error': 'Model image not found.'}), 404

    return send_from_directory(str(category_dir), safe_filename)


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

    return normalize_input_image_size(image), None

def load_uploaded_images():
    """
    Loads single or multiple uploaded files and validates them as images.
    """
    raw_files = []
    if 'files' in request.files:
        raw_files = request.files.getlist('files')
    elif 'file' in request.files:
        raw_files = [request.files['file']]

    if not raw_files:
        return None, (jsonify({'error': 'Missing image file field named "file" or "files".'}), 400)

    images = []
    for file in raw_files:
        if not file or file.filename == '':
            return None, (jsonify({'error': 'One selected file is empty.'}), 400)
        if not is_allowed_file(file.filename):
            return None, (jsonify({'error': f'Unsupported image type: {file.filename}'}), 400)

        try:
            image = Image.open(file.stream).convert('RGBA')
        except Exception:
            return None, (jsonify({'error': f'Invalid image file: {file.filename}'}), 400)

        images.append({
            'filename': file.filename,
            'image': normalize_input_image_size(image),
        })

    return images, None

def normalize_input_image_size(image):
    """
    Limits oversized input images to speed up model inference.
    """
    normalized = image.convert('RGBA')
    if max(normalized.size) <= MAX_INPUT_IMAGE_SIDE:
        return normalized

    normalized.thumbnail((MAX_INPUT_IMAGE_SIDE, MAX_INPUT_IMAGE_SIDE))
    return normalized

def send_png(image):
    img_io = io.BytesIO()
    image.save(img_io, format='PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

def image_to_data_url(image, max_side=320):
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))

    img_io = io.BytesIO()
    image.save(img_io, format='PNG')
    encoded = base64.b64encode(img_io.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'

def data_url_to_image(data_url):
    """
    Decodes a browser data URL into a PIL image.
    """
    if not data_url or ',' not in data_url:
        return None

    header, encoded = data_url.split(',', 1)
    if not header.lower().startswith('data:image/'):
        return None

    try:
        raw = base64.b64decode(unquote(encoded))
        return Image.open(io.BytesIO(raw)).convert('RGBA')
    except Exception:
        return None

def flatten_transparency(image, background=(255, 255, 255)):
    """
    FASHN VTON expects regular RGB images, so transparent wardrobe PNGs are
    composited over white like product photos.
    """
    rgba = image.convert('RGBA')
    canvas = Image.new('RGBA', rgba.size, (*background, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert('RGB')

def map_wardrobe_category_to_tryon(category, subcategory):
    if category == 'one_piece':
        return 'one-pieces'
    if category == 'upper_body':
        if subcategory == 'dress':
            return 'one-pieces'
        return 'tops'
    if category == 'lower_body':
        if subcategory in {'mini_skirt', 'maxi_skirt', 'pleated_skirt'}:
            return 'bottoms'
        return 'bottoms'
    return None

def get_tryon_pipeline():
    """
    Lazily loads FASHN VTON from the sibling Desktop folder.
    """
    global tryon_pipeline
    if tryon_pipeline is not None:
        return tryon_pipeline, None

    if not FASHN_VTON_WEIGHTS_DIR.exists():
        return None, f'FASHN VTON weights were not found at {FASHN_VTON_WEIGHTS_DIR}.'

    try:
        from fashn_vton import TryOnPipeline
    except Exception as exc:
        return None, f'FASHN VTON package is unavailable: {exc}'

    try:
        with tryon_lock:
            if tryon_pipeline is None:
                tryon_pipeline = TryOnPipeline(
                    weights_dir=str(FASHN_VTON_WEIGHTS_DIR),
                    device='cuda' if torch is not None and torch.cuda.is_available() else None,
                )
        return tryon_pipeline, None
    except Exception as exc:
        return None, f'Failed to load FASHN VTON: {exc}'

def run_tryon_image_step(
    pipeline,
    person_image,
    garment_image,
    garment_category,
    garment_subcategory,
    garment_photo_type='flat-lay',
    seed=42,
):
    tryon_category = map_wardrobe_category_to_tryon(garment_category, garment_subcategory)
    if tryon_category is None:
        return None, None, 'This wardrobe item cannot be tried on. FASHN VTON supports upper-body, lower-body, and one-piece clothing only.'

    if garment_image is None:
        return None, None, 'Missing or invalid wardrobe garment image.'

    if garment_photo_type not in {'flat-lay', 'model'}:
        return None, None, 'Unsupported garment photo type.'

    result = pipeline(
        person_image=person_image.convert('RGB'),
        garment_image=flatten_transparency(garment_image),
        category=tryon_category,
        garment_photo_type=garment_photo_type,
        num_samples=1,
        num_timesteps=TRYON_NUM_TIMESTEPS,
        guidance_scale=TRYON_GUIDANCE_SCALE,
        seed=seed,
        segmentation_free=True,
    )
    return result.images[0].convert('RGB'), tryon_category, None

def run_tryon_step(
    pipeline,
    person_image,
    garment_data_url,
    garment_category,
    garment_subcategory,
    garment_photo_type='flat-lay',
    seed=42,
):
    garment_image = data_url_to_image(garment_data_url)
    return run_tryon_image_step(
        pipeline=pipeline,
        person_image=person_image,
        garment_image=garment_image,
        garment_category=garment_category,
        garment_subcategory=garment_subcategory,
        garment_photo_type=garment_photo_type,
        seed=seed,
    )

def load_selected_person_model():
    category = request.form.get('model_category', '')
    filename = Path(request.form.get('model_filename', '')).name

    if category not in PERSON_MODEL_CATEGORIES:
        return None, 'Select a valid try-on model category first.'

    if not filename or not is_allowed_file(filename):
        return None, 'Select a valid try-on model first.'

    category_dir = (PERSON_MODELS_DIR / category).resolve()
    model_path = (category_dir / filename).resolve()
    if model_path.parent != category_dir or not model_path.exists() or not model_path.is_file():
        return None, 'Selected try-on model was not found.'

    try:
        return Image.open(model_path).convert('RGB'), None
    except Exception:
        return None, 'Selected try-on model is not a valid image.'

def create_clip_classifier():
    if torch is None or CLIPModel is None or CLIPProcessor is None:
        return None

    try:
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        model.eval()
        return {
            'processor': processor,
            'model': model,
        }
    except Exception:
        return None

def get_clip_classifier():
    global clip_classifier
    if clip_classifier is None:
        clip_classifier = create_clip_classifier()
    return clip_classifier

def get_rembg_session():
    """
    Lazily creates one reusable rembg session for faster multi-image processing.
    """
    global rembg_session
    if rembg_session is not None:
        return rembg_session
    if remove is None or new_session is None:
        return None

    try:
        rembg_session = new_session()
    except Exception:
        rembg_session = None
    return rembg_session

def get_clip_text_features(labels_to_prompts):
    """
    Caches CLIP text features to avoid repeated prompt encoding.
    """
    classifier = get_clip_classifier()
    if classifier is None:
        return None, None

    labels = list(labels_to_prompts.keys())
    prompts = tuple(labels_to_prompts[label] for label in labels)
    cache_key = '|'.join(prompts)
    if cache_key in clip_text_feature_cache:
        return labels, clip_text_feature_cache[cache_key]

    processor = classifier['processor']
    model = classifier['model']
    text_inputs = processor(
        text=list(prompts),
        return_tensors='pt',
        padding=True,
    )
    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
        )
        if hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
        elif hasattr(text_features, 'last_hidden_state'):
            text_features = text_features.last_hidden_state[:, 0, :]
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    clip_text_feature_cache[cache_key] = text_features
    return labels, text_features

def rank_labels_with_clip_batch(images, labels_to_prompts):
    """
    Runs one CLIP image forward pass for a whole image batch.
    """
    classifier = get_clip_classifier()
    if classifier is None:
        return None

    labels, text_features = get_clip_text_features(labels_to_prompts)
    if labels is None or text_features is None:
        return None

    try:
        processor = classifier['processor']
        model = classifier['model']
        resized_images = [image.convert('RGB').resize((224, 224)) for image in images]
        image_inputs = processor(images=resized_images, return_tensors='pt')

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=image_inputs['pixel_values'])
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            elif hasattr(image_features, 'last_hidden_state'):
                image_features = image_features.last_hidden_state[:, 0, :]
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            probabilities = logits.softmax(dim=1).cpu().tolist()
    except Exception:
        return None

    return [
        list(zip(labels, row_probs))
        for row_probs in probabilities
    ]

def build_fast_item_crop(input_image):
    """
    Fast crop for product photos: trims near-white background if possible.
    """
    image = input_image.convert('RGBA').copy()
    image.thumbnail((FAST_CLASSIFY_IMAGE_SIDE, FAST_CLASSIFY_IMAGE_SIDE))

    if np is None:
        return image

    rgb = np.array(image.convert('RGB'))
    mask = np.any(rgb < 245, axis=2)
    if int(mask.sum()) < 200:
        return image

    coords = np.argwhere(mask)
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    if max_x <= min_x or max_y <= min_y:
        return image

    return image.crop((int(min_x), int(min_y), int(max_x) + 1, int(max_y) + 1))

def run_high_quality_preview_job(preview_token, crop_image, predicted_category):
    """
    Background job for generating high-quality preview images.
    """
    high_preview = build_high_quality_preview(crop_image, predicted_category)
    encoded = image_to_data_url(high_preview, max_side=320)
    with preview_lock:
        preview_cache[preview_token] = {
            'ready': True,
            'preview_image': encoded,
        }

def classify_images_fast(uploaded_images, preview_mode='fast', defer_high_quality=False):
    """
    Classifies many images with one CLIP batch call for speed.
    """
    if torch is None or CLIPModel is None or CLIPProcessor is None:
        return None, 'CLIP classification model is unavailable.'

    crops = [build_fast_item_crop(item['image']) for item in uploaded_images]
    ranked_rows = rank_labels_with_clip_batch(crops, SUBCATEGORY_PROMPTS)
    if not ranked_rows:
        return None, 'CLIP classification model is unavailable.'

    classified_rows = []
    for uploaded, crop, ranked in zip(uploaded_images, crops, ranked_rows):
        ranked.sort(key=lambda result: result[1], reverse=True)
        top_subcategory, top_confidence = ranked[0]
        top_category = SUBCATEGORY_TO_CATEGORY[top_subcategory]
        classified_rows.append({
            'filename': uploaded['filename'],
            'category': top_category,
            'subcategory': top_subcategory,
            'category_confidence': round(float(top_confidence), 4),
            'subcategory_confidence': round(float(top_confidence), 4),
            'category_source': 'clip_fast',
            'crop': crop,
        })

    def render_preview_payload(row):
        if preview_mode == 'high' and not defer_high_quality:
            preview_image = build_high_quality_preview(row['crop'], row['category'])
            preview_data_url = image_to_data_url(preview_image, max_side=320)
            preview_token = None
            preview_ready = True
        elif preview_mode == 'high' and defer_high_quality:
            preview_image = build_quick_preview(row['crop'], row['category'])
            preview_data_url = image_to_data_url(preview_image, max_side=320)
            preview_token = str(uuid.uuid4())
            preview_ready = False
            with preview_lock:
                preview_cache[preview_token] = {'ready': False}
            thread = threading.Thread(
                target=run_high_quality_preview_job,
                args=(preview_token, row['crop'].copy(), row['category']),
                daemon=True,
            )
            thread.start()
        else:
            preview_image = build_quick_preview(row['crop'], row['category'])
            preview_data_url = image_to_data_url(preview_image, max_side=320)
            preview_token = None
            preview_ready = True

        return {
            'filename': row['filename'],
            'ok': True,
            'category': row['category'],
            'subcategory': row['subcategory'],
            'category_confidence': row['category_confidence'],
            'subcategory_confidence': row['subcategory_confidence'],
            'category_source': row['category_source'],
            'preview_image': preview_data_url,
            'preview_token': preview_token,
            'preview_ready': preview_ready,
        }

    worker_limit = 2 if preview_mode == 'fast' else MAX_PREVIEW_WORKERS
    worker_count = min(worker_limit, len(classified_rows))
    if worker_count <= 1:
        return [render_preview_payload(row) for row in classified_rows], None

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(render_preview_payload, classified_rows))

    return results, None

def crop_to_alpha_bounds(image, pad_ratio=0.05):
    """
    Crops RGBA image to non-transparent bounds with small padding.
    """
    alpha = np.array(image.getchannel('A'))
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    width, height = image.size
    pad_x = max(2, int((max_x - min_x + 1) * pad_ratio))
    pad_y = max(2, int((max_y - min_y + 1) * pad_ratio))

    left = max(0, min_x - pad_x)
    top = max(0, min_y - pad_y)
    right = min(width, max_x + pad_x + 1)
    bottom = min(height, max_y + pad_y + 1)
    return image.crop((left, top, right, bottom))

def build_quick_preview(crop_image, predicted_category=None):
    """
    Fast preview path for batch mode with strict latency budgets.
    """
    image = crop_image.convert('RGBA').copy()
    image.thumbnail((256, 256))

    if np is None or cv2 is None:
        return image

    rgb = np.array(image.convert('RGB'))
    height, width = rgb.shape[:2]
    if height < 16 or width < 16:
        return image

    trim_x = max(1, int(width * 0.02))
    trim_y = max(1, int(height * 0.02))
    if width - 2 * trim_x > 8 and height - 2 * trim_y > 8:
        rgb = rgb[trim_y:height - trim_y, trim_x:width - trim_x]
        height, width = rgb.shape[:2]

    edge_pixels = np.concatenate((
        rgb[0, :, :],
        rgb[-1, :, :],
        rgb[:, 0, :],
        rgb[:, -1, :],
    ), axis=0).astype(np.int16)
    bg_color = np.median(edge_pixels, axis=0)
    color_distance = np.max(np.abs(rgb.astype(np.int16) - bg_color), axis=2)
    channel_spread = rgb.max(axis=2) - rgb.min(axis=2)

    mask = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
    sure_bg = (color_distance <= 24) | ((channel_spread <= 14) & (color_distance <= 30))
    mask[sure_bg] = cv2.GC_BGD

    center_margin_x = max(8, int(width * 0.22))
    center_margin_y = max(8, int(height * 0.22))
    mask[
        center_margin_y:height - center_margin_y,
        center_margin_x:width - center_margin_x
    ] = cv2.GC_PR_FGD

    margin_x = max(6, int(width * 0.06))
    margin_y = max(6, int(height * 0.06))
    rect = (margin_x, margin_y, max(1, width - 2 * margin_x), max(1, height - 2 * margin_y))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return image

    foreground = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if int(foreground.sum()) < 80:
        return image

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return image
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    foreground = labels == largest_idx

    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[foreground] = 255
    rgba = np.dstack((rgb, alpha))
    preview = Image.fromarray(rgba, mode='RGBA')
    return crop_to_alpha_bounds(preview)

def build_high_quality_preview(crop_image, predicted_category=None):
    """
    Builds transparent preview with quality-first fallback chain.
    1) rembg (best quality in this project)
    2) lightweight GrabCut fallback
    """
    image = crop_image.convert('RGBA').copy()
    image.thumbnail((320, 320))

    if remove is not None:
        try:
            session = get_rembg_session()
            removed = remove(image, session=session) if session is not None else remove(image)
            if isinstance(removed, Image.Image):
                return crop_to_alpha_bounds(removed.convert('RGBA'))
            decoded = Image.open(io.BytesIO(removed)).convert('RGBA')
            return crop_to_alpha_bounds(decoded)
        except Exception:
            pass

    return build_quick_preview(image, predicted_category)

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

def get_alpha_connected_components(alpha_mask):
    height, width = alpha_mask.shape
    visited = np.zeros_like(alpha_mask, dtype=bool)
    components = []

    for y in range(height):
        for x in range(width):
            if visited[y, x] or alpha_mask[y, x] == 0:
                visited[y, x] = True
                continue

            stack = [(x, y)]
            visited[y, x] = True
            count = 0

            while stack:
                current_x, current_y = stack.pop()
                count += 1
                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if (
                        next_x < 0
                        or next_y < 0
                        or next_x >= width
                        or next_y >= height
                        or visited[next_y, next_x]
                    ):
                        continue

                    visited[next_y, next_x] = True
                    if alpha_mask[next_y, next_x] > 0:
                        stack.append((next_x, next_y))

            components.append(count)

    return sorted(components, reverse=True)

def build_single_item_crop(input_image):
    segmentation = predict_segmentation_map(input_image)
    masked_image = None

    if segmentation is not None:
        mask = build_mask_from_segmentation(segmentation, CLOTHING_LABEL_IDS)
        if mask is not None:
            masked_image = apply_mask_to_image(input_image, mask)

    if masked_image is None:
        if remove is None:
            return None, 'Segmentation and background removal are unavailable.', segmentation
        masked_image = remove(input_image).convert('RGBA')

    alpha = np.array(masked_image.getchannel('A'))
    total_pixels = int((alpha > 0).sum())
    if total_pixels < SINGLE_ITEM_PIXEL_MIN:
        return None, 'No valid single clothing item was detected.', segmentation

    components = get_alpha_connected_components(alpha)
    if not components:
        return None, 'No valid single clothing item was detected.', segmentation

    major_count = components[0]
    secondary_count = components[1] if len(components) > 1 else 0
    secondary_ratio = secondary_count / major_count if major_count > 0 else 0
    major_components = [count for count in components if count >= SINGLE_ITEM_PIXEL_MIN]

    if (
        len(major_components) > MAX_MAJOR_COMPONENTS
        or secondary_ratio > MAX_SECONDARY_COMPONENT_RATIO
    ):
        return None, 'Only one clothing item is allowed per upload.', segmentation

    bbox = masked_image.getbbox()
    if bbox is None:
        return None, 'No valid single clothing item was detected.', segmentation

    return masked_image.crop(bbox), None, segmentation

def rank_labels_with_clip(image, labels_to_prompts):
    classifier = get_clip_classifier()
    if classifier is None:
        return None

    processor = classifier['processor']
    model = classifier['model']
    try:
        labels, text_features = get_clip_text_features(labels_to_prompts)
        if labels is None or text_features is None:
            return None

        resized_image = image.convert('RGB').resize((224, 224))
        image_inputs = processor(
            images=resized_image,
            return_tensors='pt',
        )

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=image_inputs['pixel_values'])
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            elif hasattr(image_features, 'last_hidden_state'):
                image_features = image_features.last_hidden_state[:, 0, :]
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            probabilities = logits.softmax(dim=1).cpu().tolist()[0]
    except Exception:
        return None

    return list(zip(labels, probabilities))

def infer_major_category_from_segmentation(segmentation):
    """
    Uses parser labels as first-pass major category detection.
    """
    if segmentation is None or np is None:
        return None, 0.0

    total_pixels = int(segmentation.size)
    if total_pixels == 0:
        return None, 0.0

    category_scores = {}
    for category, label_set in SEGMENTATION_CATEGORY_LABELS.items():
        score = int(np.isin(segmentation, list(label_set)).sum())
        category_scores[category] = score

    top_category = max(category_scores, key=category_scores.get)
    top_pixels = category_scores[top_category]
    if top_pixels < SINGLE_ITEM_PIXEL_MIN:
        return None, 0.0

    confidence = float(top_pixels) / float(total_pixels)
    return top_category, confidence

def infer_lower_body_subcategory(segmentation, item_crop):
    """
    Improves lower-body accuracy using segmentation labels + silhouette ratio.
    """
    width, height = item_crop.size
    aspect_ratio = (height / width) if width > 0 else 0.0
    if segmentation is None:
        segmentation = np.zeros((1, 1), dtype=np.uint8)

    skirt_pixels = int(np.isin(segmentation, [5]).sum())
    pants_pixels = int(np.isin(segmentation, [6]).sum())

    if skirt_pixels > pants_pixels:
        if aspect_ratio < 0.95:
            return 'mini_skirt'
        if aspect_ratio > 1.2:
            return 'maxi_skirt'
        return 'pleated_skirt'

    if aspect_ratio < 1.0:
        lower_clip = rank_labels_with_clip(item_crop, {
            'casual_shorts': SUBCATEGORY_PROMPTS['casual_shorts'],
            'sports_shorts': SUBCATEGORY_PROMPTS['sports_shorts'],
        })
        if not lower_clip:
            return 'casual_shorts'
        lower_clip.sort(key=lambda result: result[1], reverse=True)
        return lower_clip[0][0]

    if aspect_ratio > 1.75:
        return 'leggings'

    lower_clip = rank_labels_with_clip(item_crop, {
        'jeans': SUBCATEGORY_PROMPTS['jeans'],
        'dress_pants': SUBCATEGORY_PROMPTS['dress_pants'],
        'sweatpants_joggers': SUBCATEGORY_PROMPTS['sweatpants_joggers'],
        'leggings': SUBCATEGORY_PROMPTS['leggings'],
    })
    if not lower_clip:
        return 'dress_pants'
    lower_clip.sort(key=lambda result: result[1], reverse=True)
    return lower_clip[0][0]

def infer_upper_body_subcategory(segmentation, item_crop):
    """
    Uses parser labels first, then a reduced CLIP label set for speed.
    """
    width, height = item_crop.size
    aspect_ratio = (height / width) if width > 0 else 0.0
    if segmentation is not None:
        dress_pixels = int(np.isin(segmentation, [4]).sum())
        upper_pixels = int(np.isin(segmentation, [3]).sum())
        if dress_pixels > upper_pixels:
            return 'dress', 0.9

    if aspect_ratio > 1.55:
        shortlist = {
            'trench_coat': SUBCATEGORY_PROMPTS['trench_coat'],
            'overcoat': SUBCATEGORY_PROMPTS['overcoat'],
            'down_jacket': SUBCATEGORY_PROMPTS['down_jacket'],
            'dress': SUBCATEGORY_PROMPTS['dress'],
        }
    elif aspect_ratio > 1.15:
        shortlist = {
            'hoodie_sweatshirt': SUBCATEGORY_PROMPTS['hoodie_sweatshirt'],
            'sweater_pullover': SUBCATEGORY_PROMPTS['sweater_pullover'],
            'cardigan': SUBCATEGORY_PROMPTS['cardigan'],
            'jacket': SUBCATEGORY_PROMPTS['jacket'],
            'suit_jacket': SUBCATEGORY_PROMPTS['suit_jacket'],
        }
    else:
        shortlist = {
            't_shirt': SUBCATEGORY_PROMPTS['t_shirt'],
            'tank_top_vest': SUBCATEGORY_PROMPTS['tank_top_vest'],
            'shirt_blouse': SUBCATEGORY_PROMPTS['shirt_blouse'],
            'polo_shirt': SUBCATEGORY_PROMPTS['polo_shirt'],
        }

    upper_rank = rank_labels_with_clip(item_crop, shortlist)
    if not upper_rank:
        fallback = next(iter(shortlist.keys()))
        return fallback, 0.55
    upper_rank.sort(key=lambda result: result[1], reverse=True)
    return upper_rank[0][0], upper_rank[0][1]

def infer_footwear_subcategory(item_crop):
    """
    Uses compact footwear candidate groups to reduce CLIP calls.
    """
    width, height = item_crop.size
    aspect_ratio = (height / width) if width > 0 else 0.0

    if aspect_ratio < 0.55:
        shortlist = {
            'sandals': SUBCATEGORY_PROMPTS['sandals'],
            'slippers': SUBCATEGORY_PROMPTS['slippers'],
            'flip_flops': SUBCATEGORY_PROMPTS['flip_flops'],
        }
    elif aspect_ratio > 1.05:
        shortlist = {
            'ankle_boots': SUBCATEGORY_PROMPTS['ankle_boots'],
            'high_boots': SUBCATEGORY_PROMPTS['high_boots'],
            'martin_boots': SUBCATEGORY_PROMPTS['martin_boots'],
        }
    else:
        shortlist = {
            'sneakers': SUBCATEGORY_PROMPTS['sneakers'],
            'skate_shoes': SUBCATEGORY_PROMPTS['skate_shoes'],
            'running_shoes': SUBCATEGORY_PROMPTS['running_shoes'],
            'oxfords': SUBCATEGORY_PROMPTS['oxfords'],
            'loafers': SUBCATEGORY_PROMPTS['loafers'],
            'derby_shoes': SUBCATEGORY_PROMPTS['derby_shoes'],
        }

    footwear_rank = rank_labels_with_clip(item_crop, shortlist)
    if not footwear_rank:
        fallback = next(iter(shortlist.keys()))
        return fallback, 0.55
    footwear_rank.sort(key=lambda result: result[1], reverse=True)
    return footwear_rank[0][0], footwear_rank[0][1]

def classify_single_item(input_image):
    if np is None:
        return None, 'NumPy is unavailable for single-item validation.'

    item_crop, crop_error, segmentation = build_single_item_crop(input_image)
    if crop_error:
        return None, crop_error

    seg_category, seg_confidence = infer_major_category_from_segmentation(segmentation)
    top_category = seg_category
    category_confidence = seg_confidence
    source = 'segmentation'

    if top_category is None:
        category_rank = rank_labels_with_clip(item_crop, CATEGORY_PROMPTS)
        if not category_rank:
            return None, 'CLIP classification model is unavailable.'
        category_rank.sort(key=lambda result: result[1], reverse=True)
        top_category, category_confidence = category_rank[0]
        source = 'clip'

    if top_category == 'lower_body':
        top_subcategory = infer_lower_body_subcategory(segmentation, item_crop)
        subcategory_confidence = 0.75
    elif top_category == 'upper_body':
        top_subcategory, subcategory_confidence = infer_upper_body_subcategory(segmentation, item_crop)
    elif top_category == 'footwear':
        top_subcategory, subcategory_confidence = infer_footwear_subcategory(item_crop)
    else:
        subcategories = CATEGORY_TAXONOMY[top_category]
        subcategory_prompt_map = {
            subcategory: SUBCATEGORY_PROMPTS[subcategory]
            for subcategory in subcategories
        }
        subcategory_rank = rank_labels_with_clip(item_crop, subcategory_prompt_map)
        if not subcategory_rank:
            return None, 'CLIP classification model is unavailable.'
        subcategory_rank.sort(key=lambda result: result[1], reverse=True)
        top_subcategory, subcategory_confidence = subcategory_rank[0]

    return {
        'category': top_category,
        'category_confidence': round(float(category_confidence), 4),
        'subcategory': top_subcategory,
        'subcategory_confidence': round(float(subcategory_confidence), 4),
        'category_source': source,
        'all_categories': CATEGORY_TAXONOMY,
        'preview_image': image_to_data_url(item_crop, max_side=320),
    }, None

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
    started_at = time.perf_counter()
    uploaded_images, error = load_uploaded_images()
    if error:
        return error

    preview_mode = request.form.get('preview_mode', 'high').strip().lower()
    if preview_mode not in {'fast', 'high'}:
        preview_mode = 'fast'

    results, classify_error = classify_images_fast(
        uploaded_images,
        preview_mode=preview_mode,
        defer_high_quality=(preview_mode == 'high'),
    )
    if classify_error:
        return jsonify({'error': classify_error}), 503

    return jsonify({
        'results': results,
        'total': len(results),
        'failed': 0,
        'success': len(results),
        'has_error': False,
        'preview_mode': preview_mode,
        'processing_ms': int((time.perf_counter() - started_at) * 1000),
        'all_categories': CATEGORY_TAXONOMY,
    })

@app.route('/extract-item', methods=['POST'])
def extract_item():
    """
    Classifies one single uploaded clothing image.
    """
    started_at = time.perf_counter()
    input_image, error = load_uploaded_image()
    if error:
        return error

    preview_mode = request.form.get('preview_mode', 'high').strip().lower()
    if preview_mode not in {'fast', 'high'}:
        preview_mode = 'high'

    results, classify_error = classify_images_fast([{
        'filename': 'single_upload',
        'image': input_image,
    }], preview_mode=preview_mode, defer_high_quality=False)
    if classify_error or not results:
        return jsonify({'error': classify_error or 'Failed to classify image.'}), 503

    return jsonify({
        'ok': True,
        **results[0],
        'preview_mode': preview_mode,
        'processing_ms': int((time.perf_counter() - started_at) * 1000),
    })

@app.route('/preview-result/<preview_token>', methods=['GET'])
def preview_result(preview_token):
    """
    Returns asynchronous high-quality preview generation result.
    """
    with preview_lock:
        data = preview_cache.get(preview_token)

    if data is None:
        return jsonify({'error': 'Preview token not found.'}), 404

    if not data.get('ready'):
        return jsonify({'ready': False})

    return jsonify({
        'ready': True,
        'preview_image': data.get('preview_image'),
    })

@app.route('/try-on', methods=['POST'])
def try_on():
    """
    Runs FASHN VTON with a bundled try-on model and a wardrobe item data URL.
    """
    pipeline, pipeline_error = get_tryon_pipeline()
    if pipeline_error:
        return jsonify({'error': pipeline_error}), 503

    person_image, model_error = load_selected_person_model()
    if model_error:
        return jsonify({'error': model_error}), 400

    try:
        started_at = time.perf_counter()
        output, tryon_category, step_error = run_tryon_step(
            pipeline=pipeline,
            person_image=person_image,
            garment_data_url=request.form.get('garment_image', ''),
            garment_category=request.form.get('category', ''),
            garment_subcategory=request.form.get('subcategory', ''),
            seed=42,
        )
        if step_error:
            return jsonify({'error': step_error}), 400
        return jsonify({
            'ok': True,
            'category': tryon_category,
            'result_image': image_to_data_url(output, max_side=None),
            'processing_ms': int((time.perf_counter() - started_at) * 1000),
        })
    except Exception as exc:
        return jsonify({'error': f'Try-on generation failed: {exc}'}), 500

@app.route('/try-on-worn-garment', methods=['POST'])
def try_on_worn_garment():
    """
    Runs FASHN VTON using a garment photo where the clothing is worn by another person.
    """
    if 'garment' not in request.files:
        return jsonify({'error': 'Missing worn garment photo field named "garment".'}), 400

    garment_file = request.files['garment']
    if not garment_file or garment_file.filename == '':
        return jsonify({'error': 'No worn garment photo selected.'}), 400

    if not is_allowed_file(garment_file.filename):
        return jsonify({'error': 'Unsupported worn garment photo type.'}), 400

    pipeline, pipeline_error = get_tryon_pipeline()
    if pipeline_error:
        return jsonify({'error': pipeline_error}), 503

    person_image, model_error = load_selected_person_model()
    if model_error:
        return jsonify({'error': model_error}), 400

    try:
        garment_image = Image.open(garment_file.stream).convert('RGB')
    except Exception:
        return jsonify({'error': 'Uploaded worn garment photo is not a valid image.'}), 400

    try:
        started_at = time.perf_counter()
        output, tryon_category, step_error = run_tryon_image_step(
            pipeline=pipeline,
            person_image=person_image,
            garment_image=garment_image,
            garment_category=request.form.get('category', ''),
            garment_subcategory=request.form.get('subcategory', ''),
            garment_photo_type='model',
            seed=44,
        )
        if step_error:
            return jsonify({'error': step_error}), 400
        return jsonify({
            'ok': True,
            'category': tryon_category,
            'garment_photo_type': 'model',
            'result_image': image_to_data_url(output, max_side=None),
            'processing_ms': int((time.perf_counter() - started_at) * 1000),
        })
    except Exception as exc:
        return jsonify({'error': f'Worn-garment try-on generation failed: {exc}'}), 500

@app.route('/try-on-outfit', methods=['POST'])
def try_on_outfit():
    """
    Runs sequential FASHN VTON steps for an upper-body + lower-body outfit.
    """
    upper_image = request.form.get('upper_image', '')
    lower_image = request.form.get('lower_image', '')
    if not upper_image and not lower_image:
        return jsonify({'error': 'Select at least one upper-body or lower-body wardrobe item.'}), 400

    pipeline, pipeline_error = get_tryon_pipeline()
    if pipeline_error:
        return jsonify({'error': pipeline_error}), 503

    current_image, model_error = load_selected_person_model()
    if model_error:
        return jsonify({'error': model_error}), 400

    try:
        started_at = time.perf_counter()
        steps = []

        if upper_image:
            current_image, tryon_category, step_error = run_tryon_step(
                pipeline=pipeline,
                person_image=current_image,
                garment_data_url=upper_image,
                garment_category='upper_body',
                garment_subcategory=request.form.get('upper_subcategory', ''),
                seed=42,
            )
            if step_error:
                return jsonify({'error': f'Upper-body try-on failed: {step_error}'}), 400
            steps.append(tryon_category)

        if lower_image:
            current_image, tryon_category, step_error = run_tryon_step(
                pipeline=pipeline,
                person_image=current_image,
                garment_data_url=lower_image,
                garment_category='lower_body',
                garment_subcategory=request.form.get('lower_subcategory', ''),
                seed=43,
            )
            if step_error:
                return jsonify({'error': f'Lower-body try-on failed: {step_error}'}), 400
            steps.append(tryon_category)

        return jsonify({
            'ok': True,
            'categories': steps,
            'result_image': image_to_data_url(current_image, max_side=None),
            'processing_ms': int((time.perf_counter() - started_at) * 1000),
        })
    except Exception as exc:
        return jsonify({'error': f'Outfit try-on generation failed: {exc}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
