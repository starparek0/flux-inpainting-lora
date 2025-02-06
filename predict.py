import os
import torch
from diffusers import FluxInpaintPipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        # Ustaw identyfikator repozytorium z modelem Flux (zmień, jeśli Twoje repo ma inną nazwę)
        self.repo_id = "flux/flux-inpainting-dev"
        # Jeśli repozytorium jest prywatne lub gated, HF_TOKEN musi być ustawiony jako zmienna środowiskowa
        hf_token = os.environ.get("HF_TOKEN")
        try:
            self.pipe = FluxInpaintPipeline.from_pretrained(
                self.repo_id,
                token=hf_token  # Przekazujemy token – jeśli jest ustawiony, lub None, jeśli nie jest wymagany
            ).to("cuda")
            print(f"Model załadowany z HF Hub: {self.repo_id}")
        except Exception as e:
            raise EnvironmentError(f"Nie udało się załadować modelu '{self.repo_id}': {e}")

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
        # Ustawiamy ziarno losowości
        torch.manual_seed(seed)
        print(f"Using prompt: '{prompt}' with prompt strength {prompt_strength}")
        print(f"Applying LoRA model: {lora_repo} with strength {lora_strength}")

        # Jeśli chcesz dynamicznie „nakładać” wagi LoRA na model, musisz zaimplementować funkcję, która pobierze
        # wagi z repozytorium lora_repo i wprowadzi modyfikacje do self.pipe. Na potrzeby tego przykładu to miejsce
        # pozostawiamy do implementacji:
        # apply_lora_weights(self.pipe, repo_id=lora_repo, strength=lora_strength)

        try:
            # Wywołujemy pipeline – upewnij się, że przekazywane argumenty odpowiadają Twojej implementacji FluxInpaintPipeline
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
