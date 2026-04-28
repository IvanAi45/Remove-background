# Clothing Extractor

This is a small school project for extracting clothing items from a person image.
The backend uses a human parsing model to keep clothing pixels, such as tops,
pants, skirts, dresses, hats, scarves, and belts, while removing background and
non-clothing body parts.

The app currently supports two output styles:

- `Extract Clothes`: keeps all detected clothing in one transparent PNG
- `Classify Single Item`: validates one uploaded clothing item and classifies it
  into upper body, lower body, or footwear with a detailed subtype

## Project Structure

```text
clothing-remover/
  backend/
    app.py
    requirements.txt
    models/
      fashn-human-parser/
  frontend/
    index.html
```

## Requirements

- Python 3.10 or newer
- Tested locally with Python 3.12
- Internet access for the first run, because the model weights are downloaded
  from Hugging Face automatically

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
processing libraries.

## Run The Backend

From the project folder, run:

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

The Flask server should start at:

```text
http://127.0.0.1:5000
```

The first image extraction or first classification may be slow because the app
may download model weights from Hugging Face on first use. After that, the
models are cached locally.

If the model has already been downloaded manually, it can be stored in:

```text
backend/models/fashn-human-parser/
```

The classification flow also uses this Hugging Face model:

```text
openai/clip-vit-base-patch32
```

## Run The Frontend

Open this file in a browser:

```text
frontend/index.html
```

Then upload an image and choose one of these actions:

- **Extract Clothes**: returns one transparent PNG containing all detected
  clothing
- **Classify Single Item**: accepts only one clothing item per upload and returns
  category + subtype classification result

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

## Notes

- The extracted result is a transparent PNG.
- The frontend displays the result on a warm wardrobe-style card background.
- This project is intended for clothing extraction, fashion organization, and
  electronic wardrobe use cases.
- It does not generate or infer hidden body content.
- `Classify Single Item` requires exactly one major clothing item in the image.
- Classification uses a zero-shot CLIP classifier; quality depends on image
  clarity, camera angle, and whether the garment occupies most of the frame.

## Model Credit

This project uses the open-source FASHN Human Parser package:

https://github.com/fashn-AI/fashn-human-parser

No paid API key is required. The model runs locally after installation. On the
first run, the model weights are downloaded automatically from Hugging Face and
cached on the user's computer.

This project also uses `rembg` for the background-removal endpoint:

https://github.com/danielgatis/rembg

This project is for school project / demo use. If the project is later used for
commercial purposes, review the model license before deployment.
