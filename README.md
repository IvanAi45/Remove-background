# Digital Wardrobe + Virtual Try-On

This is a school project for building a local digital wardrobe. It can classify
uploaded clothing images into wardrobe sections, save them in the browser, and
run virtual try-on for supported clothing items.

The project currently supports:

- Batch clothing classification into `Upper Body`, `Lower Body`, and `Footwear`
- Browser-local wardrobe storage with category shelves
- Preset try-on models grouped as `Male`, `Female`, and `Non-Binary`
- Virtual try-on from wardrobe items using FASHN VTON v1.5
- Try-on support for upper-body and lower-body garments
- Outfit try-on by selecting one upper-body item and one lower-body item

Footwear can be stored in the wardrobe, but shoe try-on is not supported by the
current FASHN VTON model.

## Project Structure

```text
clothing-remover/
  backend/
    app.py
    download_tryon_weights.py
    requirements.txt
    models/
      fashn-human-parser/
      fashn-vton-1.5/
  frontend/
    index.html
    person-models/
      Male/
      Female/
      Non-Binary/
```

## Requirements

- Python 3.10 or newer
- Tested locally with Python 3.12
- NVIDIA GPU is recommended for virtual try-on
- Internet access for first setup, because model packages and weights are
  downloaded from GitHub and Hugging Face

## Setup

Open a terminal in the project folder:

```powershell
cd C:\Users\YOUR_NAME\Desktop\Remove-background-main
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Install the backend dependencies with the virtual environment Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

The first installation can take a while because it installs PyTorch and image
processing libraries. If your team is using CUDA, install the CUDA-enabled
PyTorch wheel that matches the local machine before running try-on.

Download the FASHN VTON try-on weights into the project:

```powershell
.\.venv\Scripts\python.exe backend\download_tryon_weights.py
```

The weights are saved under:

```text
backend/models/fashn-vton-1.5/weights/
```

The `models/` directory is intentionally ignored by Git because the model files
are large.

## Run The Backend

From the project folder, run:

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

Open the app:

```text
http://127.0.0.1:5000/
```

Check backend status:

```text
http://127.0.0.1:5000/health
```

## Run The Workflow

1. Upload one or more clothing images.
2. Click `Classify Batch Images`.
3. The detected items are saved into `My Wardrobe`.
4. Choose a preset model from `Male`, `Female`, or `Non-Binary`.
5. Click `Try On` on an upper-body or lower-body wardrobe item.
6. Or click `Select Outfit` on one upper-body item and one lower-body item,
   then click `Generate Outfit Try-On`.
7. The generated result appears in the `Virtual Try-On` section.

## Run The Frontend

You normally do not need to open `frontend/index.html` directly. The Flask
backend serves it from `http://127.0.0.1:5000/`.

## API

### `POST /extract-clothes`

Form data:

- `file`: image file, such as `.png`, `.jpg`, `.jpeg`, or `.webp`

Response:

- Transparent PNG containing the extracted clothing

### `POST /extract-items`

Form data:

- `files`: one or more image files, such as `.png`, `.jpg`, `.jpeg`, or `.webp`
- `file`: still supported for backward compatibility (single-image upload)

Response:

- JSON containing batch results:
  - `results`: per-image analysis array
  - each item includes `filename`, `ok`, and either `error` or classification data
  - summary fields: `total`, `success`, `failed`, `has_error`
  - classification fields include `category`, `subcategory`, confidences, and `preview_image`

### `POST /extract-item`

Form data:

- `file`: one image file

Response:

- JSON for one file classification result with `category`, `subcategory`,
  confidences, and `preview_image`

Upper body subcategories:

- `t_shirt`
- `tank_top_vest`
- `shirt_blouse`
- `polo_shirt`
- `hoodie_sweatshirt`
- `sweater_pullover`
- `cardigan`
- `suit_jacket`
- `jacket`
- `trench_coat`
- `overcoat`
- `down_jacket`
- `dress`

Lower body subcategories:

- `jeans`
- `dress_pants`
- `sweatpants_joggers`
- `leggings`
- `casual_shorts`
- `sports_shorts`
- `mini_skirt`
- `maxi_skirt`
- `pleated_skirt`

Footwear subcategories:

- `sneakers`
- `skate_shoes`
- `running_shoes`
- `oxfords`
- `loafers`
- `derby_shoes`
- `ankle_boots`
- `high_boots`
- `martin_boots`
- `sandals`
- `slippers`
- `flip_flops`

### `POST /remove-bg`

This older endpoint removes the background only.

### `GET /person-models`

Response:

- JSON containing preset try-on model categories and image URLs.

### `POST /try-on`

Form data:

- `model_category`: preset model category, such as `Male`, `Female`, or `Non-Binary`
- `model_filename`: preset model image filename from `/person-models`
- `garment_image`: wardrobe item image as a data URL
- `category`: wardrobe category, such as `upper_body` or `lower_body`
- `subcategory`: wardrobe subcategory

Response:

- JSON with `result_image`, a data URL containing the generated try-on image

### `POST /try-on-outfit`

Form data:

- `model_category`: preset model category, such as `Male`, `Female`, or `Non-Binary`
- `model_filename`: preset model image filename from `/person-models`
- `upper_image`: optional upper-body wardrobe item image as a data URL
- `upper_subcategory`: optional upper-body subcategory
- `lower_image`: optional lower-body wardrobe item image as a data URL
- `lower_subcategory`: optional lower-body subcategory

Response:

- JSON with `result_image`, a data URL containing the generated outfit image

## Notes

- The extracted result is a transparent PNG.
- The frontend displays the result on a warm wardrobe-style card background.
- This project is intended for clothing extraction, fashion organization,
  electronic wardrobe, and virtual try-on demo use cases.
- The app uses bundled preset model photos for try-on instead of collecting
  user photos.
- It does not generate or infer hidden body content.
- `Classify Single Item` requires exactly one major clothing item in the image.
- Classification uses a zero-shot CLIP classifier; quality depends on image
  clarity, camera angle, and whether the garment occupies most of the frame.
- FASHN VTON v1.5 supports `tops`, `bottoms`, and `one-pieces`; it does not
  support footwear try-on.

## Model Credit

This project uses the open-source FASHN Human Parser package:

https://github.com/fashn-AI/fashn-human-parser

No paid API key is required. The model runs locally after installation. On the
first run, the model weights are downloaded automatically from Hugging Face and
cached on the user's computer.

The older `/remove-bg` endpoint can use `rembg` if you install it separately:

https://github.com/danielgatis/rembg

This project uses FASHN VTON v1.5 for virtual try-on:

https://github.com/fashn-AI/fashn-vton-1.5

This project is for school project / demo use. If the project is later used for
commercial purposes, review the model license before deployment.
