# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import torch
import mimetypes
import subprocess
import numpy as np
from PIL import Image
from typing import Iterator
from diffusers import FluxInpaintPipeline
from cog import BasePredictor, Input, Path
from huggingface_hub import hf_hub_download, list_repo_files  # Używamy list_repo_files

# Dodajemy typ MIME dla .webp
mimetypes.add_type("image/webp", ".webp")

MODEL_TYPE = "dev"  # "schnell" lub "dev"
MODEL_CACHE = "checkpoints"
BASE_URL = f"https://weights.replicate.delivery/default/flux-1-inpainting/{MODEL_CACHE}/"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL:", url)
    print("[~] Destination path:", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
        raise
    print("[+] Download completed in:", time.time() - start, "seconds")


def apply_lora_weights(module: torch.nn.Module, lora_state_dict: dict) -> None:
    """
    Iteruje po parametrach modułu i, jeśli w state_dict znajduje się odpowiadający klucz,
    dodaje (sumuje) wagi LoRA do istniejących wag.
    
    UWAGA: To uproszczony przykład – w praktyce integracja wag LoRA może wymagać dodatkowych operacji.
    """
    for name, param in module.named_parameters():
        if name in lora_state_dict:
            print(f"[~] Modyfikacja parametru: {name}")
            param.data += lora_state_dict[name]


def apply_lora_to_pipeline(pipe, lora_state_dict: dict) -> list:
    """
    Próbuje iterować po wszystkich parametrach pipeline (lub jego podmodułów) i modyfikuje te,
    których nazwy pasują do kluczy w lora_state_dict.
    Zwraca listę zmodyfikowanych kluczy.
    """
    updated = []
    if hasattr(pipe, "named_parameters"):
        for name, param in pipe.named_parameters():
            if name in lora_state_dict:
                print(f"[~] Modyfikacja parametru: {name}")
                param.data += lora_state_dict[name]
                updated.append(name)
    else:
        # Jeśli pipeline nie jest nn.Module, iterujemy po modułach
        for module in pipe.modules():
            for name, param in module.named_parameters(recurse=False):
                # Budujemy "pełną" nazwę – możesz ją dostosować, jeśli jest taka potrzeba
                full_name = f"{module.__class__.__name__}.{name}"
                if full_name in lora_state_dict:
                    print(f"[~] Modyfikacja parametru: {full_name}")
                    param.data += lora_state_dict[full_name]
                    updated.append(full_name)
    return updated


def load_lora_weights(pipe, lora_repo_id: str, lora_filename: str = None):
    """
    Pobiera plik z wagami LoRA z repozytorium Hugging Face i "wstrzykuje" je do parametrów pipeline,
    iterując po wszystkich parametrach.
    
    Jeśli lora_filename nie jest podany, funkcja przeszukuje repozytorium (gałąź "main")
    w poszukiwaniu dowolnego pliku z rozszerzeniem ".safetensors". Jeśli taki plik zostanie znaleziony,
    zostanie użyty; w przeciwnym razie pobierze "pytorch_model.bin".
    """
    if not lora_filename:
        try:
            file_list = list_repo_files(repo_id=lora_repo_id, revision="main")
            safetensors_files = [f for f in file_list if f.lower().endswith(".safetensors")]
            if safetensors_files:
                lora_filename = safetensors_files[0]
                print(f"[~] Znaleziono plik safetensors: {lora_filename}")
            else:
                lora_filename = "pytorch_model.bin"
                print("[~] Nie znaleziono pliku safetensors, używam 'pytorch_model.bin'")
        except Exception as e:
            print(f"[ERROR] Nie udało się pobrać listy plików z repozytorium: {e}. Używam 'pytorch_model.bin'")
            lora_filename = "pytorch_model.bin"
    
    if lora_filename.lower().endswith(".safetensors"):
        try:
            print(f"[~] Próba pobrania {lora_filename}...")
            lora_weights_path = hf_hub_download(repo_id=lora_repo_id, filename=lora_filename, revision="main")
            from safetensors.torch import load_file as safe_load
            lora_state_dict = safe_load(lora_weights_path)
        except Exception as e:
            print(f"[ERROR] Nie udało się pobrać pliku safetensors: {e}")
            raise
    else:
        try:
            print(f"[~] Próba pobrania {lora_filename}...")
            lora_weights_path = hf_hub_download(repo_id=lora_repo_id, filename=lora_filename, revision="main")
            lora_state_dict = torch.load(lora_weights_path, map_location="cpu")
        except Exception as e:
            print(f"[ERROR] Nie udało się pobrać pliku .bin: {e}")
            raise

    print("[~] Aktualny model – iteracja po parametrach:")
    updated_keys = apply_lora_to_pipeline(pipe, lora_state_dict)
    if not updated_keys:
        print("[WARNING] Żaden parametr nie został zmodyfikowany.")
    else:
        print(f"[~] Zmodyfikowano parametry: {updated_keys}")
    return pipe


class Predictor(BasePredictor):
    def setup(self) -> None:
        print(f"[~] Model type: {MODEL_TYPE}")
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)
        model_files = [f"models--black-forest-labs--FLUX.1-{MODEL_TYPE}.tar"]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = FluxInpaintPipeline.from_pretrained(
            f"black-forest-labs/FLUX.1-{MODEL_TYPE}",
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to(self.device)

    def predict(
        self,
        image: Path = Input(description="Input image for inpainting"),
        mask: Path = Input(description="Mask image"),
        prompt: str = Input(description="Text prompt for inpainting"),
        strength: float = Input(
            description="Strength of inpainting. Higher values allow for more deviation from the original image.",
            default=0.85,
            ge=0,
            le=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. More steps usually lead to a higher quality image at the expense of slower inference.",
            default=30,
            ge=1,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Guidance scale as defined in Classifier-Free Diffusion Guidance. Higher guidance scale encourages images that are closely linked to the text prompt, usually at the expense of lower image quality.",
            default=7.0,
            ge=1.0,
            le=20.0,
        ),
        height: int = Input(
            description="Height of the output image. Will be rounded to the nearest multiple of 8.",
            default=1024,
            ge=128,
            le=2048,
        ),
        width: int = Input(
            description="Width of the output image. Will be rounded to the nearest multiple of 8.",
            default=1024,
            ge=128,
            le=2048,
        ),
        num_outputs: int = Input(
            description="Number of images to generate per prompt. Batch size is set to 1",
            default=1,
            ge=1,
            le=8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        output_format: str = Input(
            description="Format of the output image",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality of the output image, from 0 to 100. 100 is best quality, 0 is lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
        lora_hf_link: str = Input(
            description="Link to HuggingFace repository containing LoRA weights (e.g. 'username/model'). Leave blank if not used.",
            default="",
        ),
    ) -> Iterator[Path]:
        if not prompt:
            raise ValueError("Please enter a text prompt.")
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"[~] Using seed: {seed}")
        height = (height + 7) // 8 * 8
        width = (width + 7) // 8 * 8
        input_image = Image.open(image).convert("RGB")
        mask_image = Image.open(mask).convert("RGB")
        if lora_hf_link.strip():
            print(f"[~] Loading LoRA weights from: {lora_hf_link}")
            self.pipe = load_lora_weights(self.pipe, lora_repo_id=lora_hf_link)
        for i in range(num_outputs):
            generator = torch.Generator(device=self.device).manual_seed(seed + i)
            result = self.pipe(
                prompt=prompt,
                image=input_image,
                mask_image=mask_image,
                height=height,
                width=width,
                strength=strength,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            extension = output_format.lower()
            extension = "jpeg" if extension == "jpg" else extension
            output_path = f"/tmp/output_{i}.{extension}"
            print(f"[~] Saving to {output_path}...")
            print(f"[~] Output format: {extension.upper()}")
            if output_format != "png":
                print(f"[~] Output quality: {output_quality}")
            save_params = {"format": extension.upper()}
            if output_format != "png":
                save_params["quality"] = output_quality
                save_params["optimize"] = True
            result.save(output_path, **save_params)
            yield Path(output_path)
