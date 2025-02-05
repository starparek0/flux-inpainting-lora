import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from diffusers import FluxInpaintPipeline

# Przykładowa funkcja ładująca i aplikująca wagi LoRA.
def apply_lora_to_model(model, lora_repo_id: str, scaling_factor: float):
    # Tutaj należy zaimplementować pobieranie wag z Hugging Face
    # oraz ich aplikację do modelu. Poniżej przykładowy komunikat.
    print(f"[INFO] Ładowanie wag LoRA z repozytorium: {lora_repo_id} (siła: {scaling_factor})")
    # Implementacja pobierania i modyfikacji wag powinna być tutaj
    return model  # Zwracamy model (w tym przykładzie bez zmian)

class Predictor(BasePredictor):
    def setup(self):
        # Ładowanie pipeline bazowego – dopasuj ścieżkę lub nazwę modelu do swoich potrzeb
        print("[INFO] Ładowanie pipeline bazowego...")
        self.pipeline = FluxInpaintPipeline.from_pretrained(
            "ścieżka/do/modelu_bazowego", revision="fp16", torch_dtype=torch.float16
        )
        self.pipeline = self.pipeline.to("cuda")
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
        base_image: Path = Input(description="Ścieżka do obrazu bazowego (RGB)"),
        mask_image: Path = Input(description="Ścieżka do obrazu maski (L – grayscale)"),
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
        # Zmiana rozmiaru obrazów
        base = base.resize((width, height))
        mask = mask.resize((width, height))

        # Zaaplikuj wagi LoRA do modelu – przykładowa implementacja
        self.pipeline.transformer = apply_lora_to_model(
            self.pipeline.transformer, lora_repo_id=lora_model, scaling_factor=lora_strength
        )

        # Opcjonalnie: modyfikacja promptu w zależności od prompt_strength
        final_prompt = prompt  # W tym przykładzie nie modyfikujemy promptu

        print(f"[INFO] Generowanie obrazu dla promptu: {final_prompt}")

        # Wywołanie pipeline – zakładamy, że pipeline obsługuje inpainting
        outputs = self.pipeline(prompt=final_prompt, image=base, mask_image=mask)
        output_image = outputs.images[0]

        # Zapisz wynikowy obraz
        output_path = Path("/tmp/output.webp")
        output_image.save(output_path, "WEBP", quality=80)
        print(f"[INFO] Obraz zapisany do: {output_path}")
        return output_path
