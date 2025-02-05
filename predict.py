import torch
from safetensors.torch import load_file  # Upewnij się, że masz zainstalowaną bibliotekę safetensors
from diffusers import FluxInpaintPipeline
# Jeśli używasz innych modeli lub pipeline, zaimportuj je odpowiednio

# Funkcja wypisująca strukturę pipeline (wszystkie atrybuty będące modułami Torch)
def print_pipeline_structure(pipe) -> None:
    print("[DEBUG] Struktura pipeline:")
    for attr in dir(pipe):
        try:
            candidate = getattr(pipe, attr)
        except Exception:
            continue
        if isinstance(candidate, torch.nn.Module):
            print(f"  Atrybut: {attr} -> {candidate.__class__.__name__}")
    print("[DEBUG] Koniec struktury pipeline.")

# Funkcja aktualizująca wagi modelu na podstawie wczytanych wag LoRA
def apply_lora_to_model(model: torch.nn.Module, lora_state_dict: dict, scaling_factor: float = 1.0) -> torch.nn.Module:
    modified = False
    # Iterujemy po parametrach modelu
    for name, param in model.named_parameters():
        # Przykładowe mapowanie kluczy:
        # Jeśli w modelu nazwa to np. "transformer_blocks.X.attn.to_k.weight",
        # zakładamy, że w pliku LoRA klucze mają postać:
        # "transformer.single_transformer_blocks.X.attn.to_k.lora_A.weight" oraz
        # "transformer.single_transformer_blocks.X.attn.to_k.lora_B.weight"
        lora_key_A = name.replace("transformer_blocks", "transformer.single_transformer_blocks") + ".lora_A.weight"
        lora_key_B = name.replace("transformer_blocks", "transformer.single_transformer_blocks") + ".lora_B.weight"
        
        if lora_key_A in lora_state_dict and lora_key_B in lora_state_dict:
            A = lora_state_dict[lora_key_A]
            B = lora_state_dict[lora_key_B]
            # Sprawdzenie zgodności wymiarów – może być konieczna modyfikacja w zależności od implementacji LoRA
            if A.shape[1] != B.shape[0]:
                print(f"[ERROR] Niezgodność wymiarów dla {name}: A.shape = {A.shape}, B.shape = {B.shape}")
                continue

            update = scaling_factor * (B @ A)
            if update.shape == param.data.shape:
                old_sum = param.data.sum().item()
                param.data.add_(update)
                new_sum = param.data.sum().item()
                print(f"[INFO] Zaktualizowano {name}: suma wag zmieniła się z {old_sum:.4f} na {new_sum:.4f}")
                modified = True
            else:
                print(f"[WARNING] Pomijam {name}: update.shape = {update.shape} != param.shape = {param.data.shape}")
    if not modified:
        print("[WARNING] Żaden parametr nie został zmodyfikowany!")
    return model

# Klasa Predictor – wymagana przez Cog
class Predictor:
    def setup(self):
        # Ładowanie pipeline z pretrenowanego modelu (dostosuj ścieżkę/model wg. swoich potrzeb)
        print("[INFO] Ładowanie pipeline...")
        try:
            self.pipeline = FluxInpaintPipeline.from_pretrained("ścieżka/do/twojego/modelu")
        except Exception as e:
            raise RuntimeError(f"Błąd przy ładowaniu pipeline: {e}")

        print_pipeline_structure(self.pipeline)
        
        # Ładowanie wag LoRA z pliku safetensors
        print("[INFO] Ładowanie wag LoRA...")
        try:
            self.lora_state_dict = load_file("prezes.safetensors")
        except Exception as e:
            raise RuntimeError(f"Błąd przy ładowaniu wag LoRA: {e}")

        # Sprawdzamy, czy pipeline ma atrybut 'transformer'
        if not hasattr(self.pipeline, "transformer"):
            raise ValueError("Pipeline nie posiada atrybutu 'transformer'. Upewnij się, że ładujesz właściwy model.")
        
        # Aktualizacja wag w części 'transformer' przy użyciu wag LoRA
        self.pipeline.transformer = apply_lora_to_model(self.pipeline.transformer, self.lora_state_dict, scaling_factor=1.0)
    
    def predict(self, prompt: str = "A high quality photorealistic image of a cat on a metallic surface") -> str:
        print(f"[INFO] Generowanie obrazu dla promptu: {prompt}")
        outputs = self.pipeline(prompt)
        # Zakładamy, że pipeline zwraca listę obrazów (np. obiektów PIL Image)
        output_image = outputs[0]
        output_path = "/tmp/output.webp"
        output_image.save(output_path, "WEBP", quality=80)
        print(f"[INFO] Obraz zapisany do {output_path}")
        return output_path
