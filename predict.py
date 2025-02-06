from cog import BasePredictor, Input, Path
from PIL import Image
import torch
# Use the Flux inpainting pipeline (which is part of the flux.1 dev codebase)
from diffusers import FluxInpaintPipeline

# =============================================================================
# Placeholder function for applying LoRA weights.
# Replace this with your actual implementation that loads the .safetensors file
# from the given Hugging Face model ID and updates the Flux pipeline accordingly.
# =============================================================================
def load_lora_weights(pipe, lora_model: str, lora_strength: float):
    print(f"Loading LoRA weights from {lora_model} with strength {lora_strength}")
    # TODO: load your LoRA weights from the model and apply them to the pipeline.
    # For example:
    #   state_dict = load_safetensors(lora_model)
    #   pipe = apply_lora_to_flux_pipeline(pipe, state_dict, strength=lora_strength)
    return pipe

# =============================================================================
# This function loads the input images, resizes them to the desired output size,
# sets up the inpainting pipeline based on Flux, applies the LoRA weights,
# and runs the pipeline to generate an output image using the given prompt.
# =============================================================================
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
    
    # Resize the base image and mask to the output dimensions
    base_img = base_img.resize((width, height))
    mask_img = mask_img.resize((width, height))
    
    # Load the Flux inpainting pipeline.
    # (Do not pass a revision parameter since we expect flux.1 dev to be the default.)
    pipe = FluxInpaintPipeline.from_pretrained(
        "flux/flux-inpainting",  # change this repo ID to your Flux inpainting model if needed
        torch_dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Apply LoRA weights (using your LoRA integration)
    pipe = load_lora_weights(pipe, lora_model, lora_strength)
    
    print(f"Generating image using prompt: '{prompt}' with guidance scale {prompt_strength}")
    
    # Run the inpainting pipeline.
    # In the Flux version, the pipeline expects:
    #   - base_img as the original image,
    #   - mask_img indicating areas to inpaint (white pixels define regions to modify),
    #   - prompt for guiding the generation,
    #   - guidance_scale (here prompt_strength) to control prompt adherence.
    output = pipe(
        prompt=prompt,
        image=base_img,
        mask_image=mask_img,
        num_inference_steps=50,
        guidance_scale=prompt_strength
    )
    
    # Return the first generated image
    return output.images[0]

# =============================================================================
# Predictor class for Cog using Flux inpainting.
# =============================================================================
class Predictor(BasePredictor):
    def setup(self):
        # One-time setup if needed.
        print("Flux model setup complete.")
    
    def predict(
        self,
        base_image: Path = Input(
            description="Upload your base image (RGB)."
        ),
        mask_image: Path = Input(
            description="Upload your mask image (RGB). White areas will be inpainted."
        ),
        lora_model: str = Input(
            description="Hugging Face LoRA model ID (e.g., shimopol/jarek)",
            default="shimopol/jarek"
        ),
        prompt: str = Input(
            description="Text prompt to guide the inpainting",
            default="A face"
        ),
        lora_strength: float = Input(
            description="Strength of the LoRA effect (0.0 - 1.0)",
            default=1.0
        ),
        prompt_strength: float = Input(
            description="Guidance scale for the prompt (higher means stronger adherence)",
            default=7.5
        ),
        height: int = Input(
            description="Output image height in pixels",
            default=512
        ),
        width: int = Input(
            description="Output image width in pixels",
            default=512
        ),
        seed: int = Input(
            description="Random seed for reproducibility",
            default=42
        )
    ) -> Path:
        # Open the base and mask images and ensure they are in RGB
        base_img = Image.open(base_image).convert("RGB")
        mask_img = Image.open(mask_image).convert("RGB")
        
        print(f"Using prompt: '{prompt}' with prompt strength {prompt_strength}")
        print(f"Applying LoRA model: {lora_model} with strength {lora_strength}")
        
        # Generate the output image
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
