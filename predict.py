from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from diffusers import FluxInpaintPipeline

# ------------------------------------------------------------------------------
# This function is a placeholder for applying LoRA weights.
# Replace its implementation with your actual logic for loading and merging
# LoRA weights into your pipeline.
# ------------------------------------------------------------------------------
def load_lora_weights(pipe, lora_model: str, lora_strength: float):
    print(f"Loading LoRA weights from {lora_model} with strength {lora_strength}")
    # TODO: Add your implementation for loading the LoRA safetensors and applying them.
    return pipe

# ------------------------------------------------------------------------------
# Generate an inpainted image using the Flux-based inpainting pipeline.
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
    
    # Resize the base and mask images to the desired output dimensions
    base_img = base_img.resize((width, height))
    mask_img = mask_img.resize((width, height))
    
    # Specify the model id. If you are using Flux 1 dev, adjust the id accordingly.
    # For example, if the correct repository is "flux/flux-inpainting-dev", change it here.
    model_id = "flux/flux-inpainting"  # Adjust this if needed.
    
    # Load the Flux inpainting pipeline.
    # Note: We pass `use_auth_token=True` so that if the model is private/gated,
    # authentication (e.g. via HF_HUB_TOKEN) is used.
    pipe = FluxInpaintPipeline.from_pretrained(
        model_id,
        local_files_only=False,
        torch_dtype=torch.float16,
        use_auth_token=True
    )
    
    # Move the pipeline to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Apply LoRA weights to the pipeline (this function must be implemented)
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
    
    # Return the first generated image
    return output.images[0]

# ------------------------------------------------------------------------------
# Predictor class for Cog using Flux inpainting.
# ------------------------------------------------------------------------------
class Predictor(BasePredictor):
    def setup(self):
        # One-time setup if needed.
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
            description="Hugging Face LoRA model ID (for example, 'shimopol/jarek').",
            default="shimopol/jarek"
        ),
        prompt: str = Input(
            description="Text prompt to guide the inpainting process.",
            default="A face"
        ),
        lora_strength: float = Input(
            description="Strength of the LoRA effect (0.0 to 1.0).",
            default=1.0
        ),
        prompt_strength: float = Input(
            description="Guidance scale for the text prompt (higher means stronger influence).",
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
        # Open base and mask images and convert them to RGB
        base_img = Image.open(base_image).convert("RGB")
        mask_img = Image.open(mask_image).convert("RGB")
        
        print(f"Using prompt: '{prompt}' with prompt strength {prompt_strength}")
        print(f"Applying LoRA model: {lora_model} with strength {lora_strength}")
        
        # Generate the output image using the inpainting function
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
        
        # Save the generated image to a temporary file in WEBP format
        output_path = "/tmp/output_0.webp"
        output_img.save(output_path, "WEBP")
        print(f"Output image saved to {output_path}")
        
        return Path(output_path)
