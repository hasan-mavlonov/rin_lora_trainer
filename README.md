# Rin LoRA Trainer

A clean, modern Stable Diffusion XL LoRA training pipeline powered by `diffusers`.

## Environment

Use Python 3.10 and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`datasets` 2.14 relies on the legacy `pyarrow.PyExtensionType` API, so `pyarrow` is pinned below 14.0 in the dependency lists. `torchvision` is pinned to match `torch` so CUDA wheels stay aligned for `xformers` and `diffusers`, and `diffusers` is kept at 0.32.x or newer so SDXL refiner components import correctly. Reinstall after pulling the latest `requirements.txt` or `environment.yaml` to avoid import errors during training.

Conda users can create an environment with CUDA 12.x builds of PyTorch:

```bash
conda env create -f environment.yaml
conda activate rin-lora
```

## Preparing data

Place your training images and captions in a folder, e.g. `data/train`. The script expects images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`) with captions stored either in sidecar `.txt` files (handled by `datasets` imagefolder) or in a `metadata.jsonl` file when using `datasets` defaults. The folder path is passed via `--train_data_dir`.

## Running training

A convenience wrapper is provided:

```bash
./train.sh \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --train-dir data/train \
  --output /output/lora-run \
  --batch-size 1 \
  --steps 1500 \
  --learning-rate 1e-4
```

The script defaults to mixed precision, gradient accumulation, checkpointing, and LoRA rank configuration but can be overridden via CLI arguments. To resume, provide `--resume_from_checkpoint` pointing to a saved checkpoint directory.

## Using the LoRA

After training, LoRA weights are written to the output directory. Load them in WebUI or ComfyUI by copying the safetensor files into your LoRA folder and selecting them alongside the SDXL base (and optional refiner) checkpoint.

## Notes

* Checkpoints and logs are saved under the configured `--output_dir` (default `/output`).
* Validation prompts can be enabled with `--validation_prompt` and optionally refined with `--pretrained_refiner_model_name_or_path`.
* The pipeline prints accelerator state and training sample counts for easier debugging.
