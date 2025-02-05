import torch
from PIL import Image
from cog import BasePredictor, Input, Path, InputImage
from diffusers import FluxInpaintPipeline

# Przykładowa funkcja ładująca i aplikująca wagi LoRA.
# W praktyce musisz zaimplementować właściwą logikę – np. pobierając wagę
# z Hugging Face i modyfikując odpowiednie warstwy modelu.
def apply_lora_to_model(model, lora_repo_id: str, scaling_factor: float):
    # Przykładowa implementacja:
    # Tu można użyć biblioteki huggingface_hub do pobrania pliku z wagami,
    # a następnie wczytać je (np. przy użyciu safetensors) i zmodyfikować model.
    # Na potrzeby tego przykładu zakładamy, że funkcja zwraca model "zmodyfikowany".
    print(f"[INFO] Ładowanie wag LoRA z repozytorium: {lora_repo_id} (siła: {scaling_factor})")
    # ... (tutaj kod pobierający i aplikujący wagi LoRA)
    return model  # zwracamy model po modyfikacji

class Predictor(BasePredictor):
    def setup(self):
        # Załaduj bazowy pipeline – ścieżkę do modelu bazowego dopasuj do swoich potrzeb
        print("[INFO] Ładowanie pipeline bazowego...")
        self.pipeline = FluxInpaintPipeline.from_pretrained(
            "ścieżka/do/modelu_bazowego", revision="fp16", torch_dtype=torch.float16
        )
        self.pipeline = self.pipeline.to("cuda")
        # Jeśli chcesz wyświetlić strukturę pipeline, możesz dodać funkcję pomocniczą:
        self.print_pipeline_structure(self.pipeline)

    def print_pipeline_structure(self, pipe):
        print("[DEBUG] Struktura pipeline:")
        for attr in dir(pipe):
            try:
                candidate = getattr(pipe, attr)
            except Exception:
                continue
            if hasattr(candidate, "__class__"):
                print(f"  Atrybut: {attr} -> {candidate.__class__.__name__}")
        print("[DEBUG] Koniec struktury pipeline.")

    def predict(
        self,
        base_image: InputImage = Input(description="Obraz bazowy (RGB)"),
        mask_image: InputImage = Input(description="Obraz maski (L – grayscale)"),
        lora_model: str = Input(
            description="Identyfikator repozytorium modelu LoRA na Hugging Face", 
            default="shimopol/prezes"
        ),
        prompt: str = Input(
            description="Tekst opisujący wygenerowany obraz", 
            default="A beautiful photorealistic scene of a futuristic cityscape"
        ),
        lora_strength: float = Input(
            description="Siła użycia wag LoRA", ge=0.0, le=2.0, default=1.0
        ),
        prompt_strength: float = Input(
            description="Siła wpływu promptu", ge=0.0, le=2.0, default=1.0
        ),
        width: int = Input(
            description="Szerokość obrazu wyjściowego", ge=64, le=2048, default=512
        ),
        height: int = Input(
            description="Wysokość obrazu wyjściowego", ge=64, le=2048, default=512
        ),
        seed: int = Input(
            description="Ziarno (seed) dla deterministycznej generacji", default=42
        )
    ) -> Path:
        # Ustawienie ziarna
        torch.manual_seed(seed)
        print(f"[INFO] Używany seed: {seed}")

        # Otwórz obrazy wejściowe
        base = Image.open(base_image).convert("RGB")
        mask = Image.open(mask_image).convert("L")
        # Zmiana rozmiaru obrazów zgodnie z ustawieniami
        base = base.resize((width, height))
        mask = mask.resize((width, height))

        # Zaaplikuj wagi LoRA do modelu (przykład – dostosuj do swojej implementacji)
        self.pipeline.transformer = apply_lora_to_model(
            self.pipeline.transformer, lora_repo_id=lora_model, scaling_factor=lora_strength
        )

        # Opcjonalnie: zmodyfikuj prompt w zależności od prompt_strength
        # (tu możesz na przykład modyfikować prompt lub warstwy modelu odpowiedzialne za tekst)
        final_prompt = prompt  # W tym przykładzie nie modyfikujemy promptu

        print(f"[INFO] Generowanie obrazu dla promptu: {final_prompt}")

        # Wywołanie pipeline – przyjmujemy, że pipeline obsługuje inpainting z podanym obrazem bazowym i maską
        outputs = self.pipeline(prompt=final_prompt, image=base, mask_image=mask)
        output_image = outputs.images[0]

        # Zapisz wynikowy obraz
        output_path = Path("/tmp/output.webp")
        output_image.save(output_path, "WEBP", quality=80)
        print(f"[INFO] Obraz zapisany do: {output_path}")
        return output_path
