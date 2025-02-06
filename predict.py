from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from diffusers import FluxInpaintPipeline

# ------------------------------------------------------------------------------
# Placeholder function for applying LoRA weights.
# Replace the content of this function with your actual logic for loading
# and merging LoRA weights into the pipeline.
# ------------------------------------------------------------------------------
def load_lora_weights(pipe, lora_model: str, lora_strength: float):
    print(f"Loading LoRA weights from {lora_model} with strength {lora_strength}")
    # TODO: Insert your code here to load the safetensors file from lora_model
    # and merge them into the pipeline weights.
    return pipe

# ------------------------------------------------------------------------------
# Generate an inpainted image using the Flux inpainting pipeline.
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
    
    # Resize the base and mask images to the desired dimensions
    base_img = base_img.resize((width, height))
    mask_img = mask_img.resize((width, height))
    
    # Use a valid model ID (do not include any "./" or forbidden characters)
    model_id = "flux/flux-inpainting"  # Adjust this if your model repo is named differently.
    
    # Load the Flux inpainting pipeline.
    # use_auth_token=True ensures that, if the model is gated or private, authentication is used.
    pipe = FluxInpaintPipeline.from_pretrained(
        model_id,
        local_files_only=False,
        torch_dtype=torch.float16,
        use_auth_token=True
    )
    
    # Move the pipeline to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Apply LoRA weights (your implementation must go inside load_lora_weights)
    pipe = load_lora_weights(pipe, lora_model, lora_strength)
    
    print(f"Generating image using prompt: '{prompt}' with guidance scale {prompt_strength}")
    
    # Run the inpainting pipeline with the provided prompt and images
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
        # One-time setup actions if needed.
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
        # Open the base and mask images and convert them to RGB
        base_img = Image.open(base_image).convert("RGB")
        mask_img = Image.open(mask_image).convert("RGB")
        
        print(f"Using prompt: '{prompt}' with prompt strength {prompt_strength}")
        print(f"Applying LoRA model: {lora_model} with strength {lora_strength}")
        
        # Generate the inpainted image
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
        
        # Save the generated image to a temporary file (WEBP format)
        output_path = "/tmp/output_0.webp"
        output_img.save(output_path, "WEBP")
        print(f"Output image saved to {output_path}")
        
        return Path(output_path)
