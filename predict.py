from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from diffusers import FluxInpaintPipeline

# ------------------------------------------------------------------------------
# Placeholder for merging LoRA weights.
# Replace this function with your actual logic for loading and applying LoRA weights.
# ------------------------------------------------------------------------------
def load_lora_weights(pipe, lora_model: str, lora_strength: float):
    print(f"Loading LoRA weights from {lora_model} with strength {lora_strength}")
    # TODO: Implement your LoRA weights merging here.
    return pipe

# ------------------------------------------------------------------------------
# Function to generate an inpainted image using FluxInpaintPipeline.
# ------------------------------------------------------------------------------
def generate_image(
    base_img: Image.Image,
    mask_img: Image.Image,
    lora_model: str,
    prompt: str,
    lora_strength: float,
    prompt_strength: float,
    height: int,
    width: int,
    seed: int
) -> Image.Image:
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    
    # Resize images to desired dimensions
    base_img = base_img.resize((width, height))
    mask_img = mask_img.resize((width, height))
    
    # IMPORTANT: Use the correct model repository ID.
    # Here we use "flux/flux-inpainting-dev" (adjust as needed).
    model_id = "flux/flux-inpainting-dev"
    
    try:
        pipe = FluxInpaintPipeline.from_pretrained(
            model_id,
            local_files_only=False,  # Set to True if you want to load only from local cache.
            torch_dtype=torch.float16,
            use_auth_token=True       # Requires HF_HUB_TOKEN to be set in your environment.
        )
    except Exception as e:
        print(f"Error loading model from '{model_id}': {e}")
        raise e

    # Move pipeline to the appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Apply LoRA weights (if implemented)
    pipe = load_lora_weights(pipe, lora_model, lora_strength)
    
    print(f"Generating image using prompt: '{prompt}' with guidance scale {prompt_strength}")
    
    # Run the inpainting pipeline
    output = pipe(
        prompt=prompt,
        image=base_img,
        mask_image=mask_img,
        num_inference_steps=50,
        guidance_scale=prompt_strength
    )
    
    return output.images[0]

# ------------------------------------------------------------------------------
# Predictor class for Cog using Flux inpainting.
# ------------------------------------------------------------------------------
class Predictor(BasePredictor):
    def setup(self):
        print("Flux inpainting model setup complete.")
    
    def predict(
        self,
        base_image: Path = Input(
            description="Upload your base image (RGB)."
        ),
        mask_image: Path = Input(
            description="Upload your mask image (RGB). White areas indicate regions to inpaint."
        ),
        lora_model: str = Input(
            description="Hugging Face LoRA model ID (e.g. 'shimopol/jarek').",
            default="shimopol/jarek"
        ),
        prompt: str = Input(
            description="Text prompt to guide the inpainting.",
            default="A face"
        ),
        lora_strength: float = Input(
            description="Strength of the LoRA effect (0.0 to 1.0).",
            default=1.0
        ),
        prompt_strength: float = Input(
            description="Guidance scale for the prompt (higher = stronger influence).",
            default=7.5
        ),
        height: int = Input(
            description="Output image height in pixels.",
            default=512
        ),
        width: int = Input(
            description="Output image width in pixels.",
            default=512
        ),
        seed: int = Input(
            description="Random seed for reproducibility.",
            default=42
        )
    ) -> Path:
        # Open and convert images to RGB
        base_img = Image.open(base_image).convert("RGB")
        mask_img = Image.open(mask_image).convert("RGB")
        
        print(f"Using prompt: '{prompt}' with prompt strength {prompt_strength}")
        print(f"Applying LoRA model: {lora_model} with strength {lora_strength}")
        
        output_img = generate_image(
            base_img,
            mask_img,
            lora_model,
            prompt,
            lora_strength,
            prompt_strength,
            height,
            width,
            seed
        )
        
        # Save the output image
        output_path = "/tmp/output_0.webp"
        output_img.save(output_path, "WEBP")
        print(f"Output image saved to {output_path}")
        
        return Path(output_path)
