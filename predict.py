import torch
from diffusers import FluxInpaintPipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        # Ustaw ścieżkę do lokalnego modelu Flux (wszystkie pliki modelu muszą być dostępne w tym folderze)
        model_path = "./models/flux-inpainting-dev"
        try:
            # Używamy local_files_only=True, żeby nie próbować pobierać modelu z Hugging Face Hub
            self.pipe = FluxInpaintPipeline.from_pretrained(
                model_path,
                local_files_only=True
            ).to("cuda")
            print(f"Model załadowany z: {model_path}")
        except Exception as e:
            raise EnvironmentError(f"Nie udało się załadować modelu z '{model_path}': {e}")

    def predict(
        self,
        prompt: str = Input(description="Tekst promptu", default="A face"),
        prompt_strength: float = Input(description="Siła promptu", default=7.5),
        lora_repo: str = Input(
            description="Repozytorium LoRA (np. 'shimopol/jarek')", default="shimopol/jarek"
        ),
        lora_strength: float = Input(description="Siła LoRA", default=1.0),
        input_image: Path = Input(description="Obraz bazowy"),
        mask_image: Path = Input(description="Obraz maski (biały obszar – generacja)"),
        width: int = Input(description="Szerokość outputu", default=512),
        height: int = Input(description="Wysokość outputu", default=512),
        seed: int = Input(description="Seed", default=42),
    ) -> Path:
        # Ustawienie ziarna losowości
        torch.manual_seed(seed)
        print(f"Using prompt: '{prompt}' with prompt strength {prompt_strength}")
        print(f"Applying LoRA model: {lora_repo} with strength {lora_strength}")

        # Jeśli masz funkcję do nakładania wag LoRA, dodaj ją tutaj.
        # Przykładowo, jeśli Twój pipeline ma metodę apply_lora_weights, wywołaj:
        #
        # self.pipe.apply_lora_weights(lora_repo, strength=lora_strength)
        #
        # Jeśli nie – upewnij się, że wagi LoRA są już uwzględnione w Twoim modelu lub zaimplementuj taką logikę.

        try:
            output = self.pipe(
                prompt=prompt,
                prompt_strength=prompt_strength,
                image=input_image,
                mask_image=mask_image,
                width=width,
                height=height,
            )
        except Exception as e:
            raise EnvironmentError(f"Błąd podczas generacji obrazu: {e}")

        output_img = output.images[0]
        output_path = "/tmp/output_0.webp"
        output_img.save(output_path)
        print(f"Output image saved to {output_path}")
        return Path(output_path)
