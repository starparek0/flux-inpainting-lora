from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

# =============================================================================
# Replace the code in this function with your actual LoRA integration.
# For example, load the LoRA .safetensors weights from the given model ID
# and update the pipeline model parameters accordingly.
# =============================================================================
def load_lora_weights(pipe, lora_model: str, lora_strength: float):
    print(f"Loading LoRA weights from {lora_model} with strength {lora_strength}")
    # Example (dummy): In your implementation, download the safetensors file from Hugging Face
    # and apply the weights to the pipeline’s underlying model.
    #
    # Example pseudocode:
    #   state_dict = load_safetensors(lora_model)  # your function to load the weights
    #   pipe.unet = apply_lora(pipe.unet, state_dict, strength=lora_strength)
    #
    # For now, we simply return the unchanged pipeline.
    return pipe

# =============================================================================
# This function loads the input images, resizes them to the desired output size,
# sets up the inpainting pipeline, applies the LoRA weights, and finally runs the
# pipeline to generate an output image using the given prompt.
# =============================================================================
def generate_image(base_img, mask_img, lora_model, prompt, lora_strength, prompt_strength, height, width, seed):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Resize base image and mask to the output dimensions
    base_img = base_img.resize((width, height))
    mask_img = mask_img.resize((width, height))
    
    # Load the inpainting pipeline.
    # (Here we use the stable-diffusion-inpainting model from RunwayML.)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Apply LoRA weights (replace the dummy implementation with your actual code)
    pipe = load_lora_weights(pipe, lora_model, lora_strength)
    
    print(f"Generating image using prompt: '{prompt}'")
    
    # Run the inpainting pipeline. The model uses:
    #  - base_img as the starting image,
    #  - mask_img to determine which areas to modify (white areas will be inpainted),
    #  - prompt to guide the generation,
    #  - guidance_scale (here prompt_strength) controls how strongly the prompt is followed.
    output = pipe(
        prompt=prompt,
        image=base_img,
        mask_image=mask_img,
        num_inference_steps=50,
        guidance_scale=prompt_strength
    )
    
    # Return the generated image (first image from the pipeline's output list)
    return output.images[0]

# =============================================================================
# Predictor class for Cog
# =============================================================================
class Predictor(BasePredictor):
    def setup(self):
        # Any one-time model setup can be performed here.
        print("Model setup complete.")
    
    def predict(
        self,
        base_image: Path = Input(
            description="Upload your base image (RGB)."
        ),
        mask_image: Path = Input(
            description="Upload your mask image (RGB). White areas will be inpainted with new content."
        ),
        lora_model: str = Input(
            description="Hugging Face LoRA model ID (e.g. shimopol/prezes)",
            default="shimopol/prezes"
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
            description="Guidance scale for the prompt (e.g. higher means stronger adherence to prompt)",
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
        # Load and convert the input images
        base_img = Image.open(base_image).convert("RGB")
        mask_img = Image.open(mask_image).convert("RGB")
        
        print(f"Using prompt: '{prompt}' with prompt strength {prompt_strength}")
        print(f"Applying LoRA model {lora_model} with strength {lora_strength}")
        
        # Generate the output image using the provided parameters.
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
        
        # Save the generated image (using WEBP format)
        output_path = "/tmp/output_0.webp"
        output_img.save(output_path, "WEBP")
        print(f"Output image saved to {output_path}")
        
        return Path(output_path)
