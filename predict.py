import torch
from diffusers import FluxInpaintPipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    @classmethod
    def setup(cls):
        # Możesz wykonać wstępne ustawienia tutaj, jeśli potrzebujesz.
        pass

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(default="A face", description="Tekst promptu"),
        prompt_strength: float = Input(default=7.5, description="Siła promptu (guidance scale)"),
        input_image: Path = Input(description="Obraz wejściowy (base image)"),
        mask_image: Path = Input(description="Obraz maski – białe obszary, gdzie ma być zmiana"),
        lora_model: str = Input(default="shimopol/jarek", description="Repozytorium modelu LoRA"),
        lora_strength: float = Input(default=1.0, description="Siła modelu LoRA"),
        inpaint_repo: str = Input(
            default="flux/flux-inpainting-dev",
            description="Repozytorium modelu inpaint (np. flux/flux-inpainting-dev)"
        ),
        output_width: int = Input(default=512, description="Szerokość obrazu wyjściowego"),
        output_height: int = Input(default=512, description="Wysokość obrazu wyjściowego"),
        seed: int = Input(default=42, description="Seed"),
        hf_token: str = Input(default="", description="Token Hugging Face (jeśli wymagany)"),
        local_files_only: bool = Input(default=False, description="Używać wyłącznie lokalnych plików?")
    ) -> Path:
        # Ustawienie seeda dla generatora
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Próba załadowania modelu inpaint z podanego repozytorium
        try:
            pipe = FluxInpaintPipeline.from_pretrained(
                inpaint_repo,
                torch_dtype=torch.float16,
                token=hf_token if hf_token else None,
                local_files_only=local_files_only
            )
        except Exception as e:
            raise EnvironmentError(f"Nie udało się załadować modelu '{inpaint_repo}': {e}")
        
        # Jeśli podano model LoRA – próba jego zastosowania
        if lora_model:
            try:
                # UWAGA: Upewnij się, że Twoja wersja pipeline posiada metodę apply_lora.
                pipe.apply_lora(lora_model, strength=lora_strength)
            except Exception as e:
                # Jeśli nie uda się zastosować LoRA, wyświetl ostrzeżenie, ale nie przerywaj pracy.
                print(f"Ostrzeżenie: Nie udało się zastosować LoRA '{lora_model}': {e}")
        
        # Wywołanie pipeline – przekazujemy prompt, obrazy oraz parametry generacji
        output = pipe(
            prompt=prompt,
            image=input_image,
            mask_image=mask_image,
            height=output_height,
            width=output_width,
            num_inference_steps=50,
            guidance_scale=prompt_strength,
            generator=generator
        )
        
        # Zapisanie obrazu wyjściowego do pliku WEBP
        output_path = Path("/tmp/output_0.webp")
        output.images[0].save(output_path)
        return output_path
