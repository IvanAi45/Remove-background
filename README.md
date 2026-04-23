# Clothing Extractor

This is a small school project for extracting clothing items from a person image.
The backend uses a human parsing model to keep clothing pixels, such as tops,
pants, skirts, dresses, hats, scarves, and belts, while removing background and
non-clothing body parts.

## Project Structure

```text
clothing-remover/
  backend/
    app.py
    requirements.txt
  frontend/
    index.html
```

## Requirements

- Python 3.10 or newer
- Internet access for the first run, because the model weights are downloaded
  from Hugging Face automatically

## Setup

Open a terminal in the project folder:

```powershell
cd C:\Users\YOUR_NAME\Desktop\clothing-remover
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
pip install -r backend\requirements.txt
```

The first installation can take a while because it installs PyTorch and image
processing libraries.

## Run The Backend

From the project folder, run:

```powershell
python backend\app.py
```

The Flask server should start at:

```text
http://127.0.0.1:5000
```

The first image extraction may be slow because `fashn-human-parser` downloads
the model weights from Hugging Face. After that, the model is cached locally.

## Run The Frontend

Open this file in a browser:

```text
frontend/index.html
```

Then upload a person image and click **Extract Clothes**.

## API

### `POST /extract-clothes`

Form data:

- `file`: image file, such as `.png`, `.jpg`, `.jpeg`, or `.webp`

Response:

- Transparent PNG containing the extracted clothing

### `POST /remove-bg`

This older endpoint removes the background only.

## Notes

- The extracted result is a transparent PNG.
- The frontend displays the result on a warm wardrobe-style card background.
- This project is intended for clothing extraction, fashion organization, and
  electronic wardrobe use cases.
- It does not generate or infer hidden body content.

## Model Credit

This project uses the open-source FASHN Human Parser package:

https://github.com/fashn-AI/fashn-human-parser

No paid API key is required. The model runs locally after installation. On the
first run, the model weights are downloaded automatically from Hugging Face and
cached on the user's computer.

This project is for school project / demo use. If the project is later used for
commercial purposes, review the model license before deployment.
