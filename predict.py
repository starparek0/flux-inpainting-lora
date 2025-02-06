import torch
from diffusers import FluxInpaintPipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    @classmethod
    def setup(cls):
        # Jeśli chcesz, możesz tu wykonać wstępne ustawienia,
        # ale kluczowe elementy ładowania modelu wykonamy w metodzie predict.
        return

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(default="A face", description="Tekst promptu"),
        prompt_strength: float = Input(default=7.5, description="Siła promptu (guidance scale)"),
        input_image: Path = Input(description="Obraz wejściowy (base image)"),
        mask_image: Path = Input(description="Obraz maski – białe pole tam, gdzie mają być zmiany"),
        lora_model: str = Input(default="shimopol/prezes", description="Repozytorium LoRA"),
        lora_strength: float = Input(default=1.0, description="Siła modelu LoRA"),
        output_width: int = Input(default=512, description="Szerokość obrazu wyjściowego"),
        output_height: int = Input(default=512, description="Wysokość obrazu wyjściowego"),
        seed: int = Input(default=42, description="Seed"),
        hf_token: str = Input(default="", description="Hugging Face token (jeśli wymagany)")
    ) -> Path:
        # Ustawienie seeda
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Upewnij się, że repozytorium modelu inpaint jest poprawne.
        # Jeśli korzystasz z publicznego modelu, zmień poniżej na np. "flux/flux-inpainting".
        repo_id = "flux/flux-inpainting"  # <- modyfikuj, jeśli potrzebujesz innej wersji

        try:
            pipe = FluxInpaintPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                token=hf_token if hf_token != "" else None
            )
        except Exception as e:
            raise EnvironmentError(f"Nie udało się załadować modelu '{repo_id}': {e}")

        # Jeśli chcesz zastosować LoRA, zakładamy, że pipeline ma metodę apply_lora.
        # Jeśli nie, musisz wprowadzić odpowiednie modyfikacje.
        if lora_model:
            try:
                # Przykładowe wywołanie – upewnij się, że taka funkcja istnieje w Twojej implementacji.
                pipe.apply_lora(lora_model, strength=lora_strength)
            except Exception as e:
                # Możesz też po prostu wypisać ostrzeżenie, jeśli LoRA nie jest dostępna.
                print(f"Ostrzeżenie: Nie udało się zastosować LoRA '{lora_model}': {e}")

        # Uruchomienie pipeline – przekazujemy prompt, obraz wejściowy, maskę oraz pozostałe ustawienia.
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

        # Zapisujemy wynik do pliku WEBP
        output_path = Path("/tmp/output_0.webp")
        output.images[0].save(output_path)
        return output_path
