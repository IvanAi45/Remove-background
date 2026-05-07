#!/usr/bin/env python3
"""Download FASHN VTON v1.5 weights for the integrated wardrobe demo."""

from pathlib import Path

from huggingface_hub import hf_hub_download


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "models" / "fashn-vton-1.5" / "weights"


def download_tryon_model() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading FASHN VTON model.safetensors...")
    hf_hub_download(
        repo_id="fashn-ai/fashn-vton-1.5",
        filename="model.safetensors",
        local_dir=str(WEIGHTS_DIR),
    )


def download_dwpose_models() -> None:
    dwpose_dir = WEIGHTS_DIR / "dwpose"
    dwpose_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]:
        print(f"Downloading DWPose/{filename}...")
        hf_hub_download(
            repo_id="fashn-ai/DWPose",
            filename=filename,
            local_dir=str(dwpose_dir),
        )


def download_human_parser() -> None:
    print("Preparing FASHN Human Parser cache...")
    from fashn_human_parser import FashnHumanParser

    FashnHumanParser(device="cpu")


def main() -> None:
    download_tryon_model()
    download_dwpose_models()
    download_human_parser()
    print(f"Done. Try-on weights are in: {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
