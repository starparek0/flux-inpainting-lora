import torch
from safetensors.torch import load_file  # Upewnij się, że masz zainstalowaną bibliotekę safetensors

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
        # Przykładowe mapowanie klucza:
        # Jeśli w modelu nazwa to "transformer_blocks.X.attn.to_k.weight",
        # to zakładamy, że w LoRA mamy klucze:
        # "transformer.single_transformer_blocks.X.attn.to_k.lora_A.weight" oraz
        # "transformer.single_transformer_blocks.X.attn.to_k.lora_B.weight"
        lora_key_A = name.replace("transformer_blocks", "transformer.single_transformer_blocks") + ".lora_A.weight"
        lora_key_B = name.replace("transformer_blocks", "transformer.single_transformer_blocks") + ".lora_B.weight"
        
        if lora_key_A in lora_state_dict and lora_key_B in lora_state_dict:
            A = lora_state_dict[lora_key_A]
            B = lora_state_dict[lora_key_B]
            # Zakładamy, że operacja B @ A daje tensor o kształcie zgodnym z parametrem
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

if __name__ == '__main__':
    # Ładowanie pipeline – przykładowo przy użyciu FluxInpaintPipeline
    try:
        from diffusers import FluxInpaintPipeline
        pipeline = FluxInpaintPipeline.from_pretrained("ścieżka/do/twojego/modelu")
    except Exception as e:
        print("Błąd przy ładowaniu modelu:", e)
        exit(1)

    # Wypisanie struktury pipeline dla debugowania
    print_pipeline_structure(pipeline)

    # Ładowanie wag LoRA z pliku safetensors
    try:
        lora_state_dict = load_file("prezes.safetensors")
    except Exception as e:
        print("Błąd przy ładowaniu wag LoRA:", e)
        exit(1)

    # Sprawdzamy, czy pipeline ma atrybut 'transformer'
    if not hasattr(pipeline, "transformer"):
        print("Pipeline nie posiada atrybutu 'transformer'. Upewnij się, że ładujesz właściwy model.")
        exit(1)
    
    # (Opcjonalnie) Zapisujemy oryginalny stan wag części 'transformer' do porównania
    original_transformer_state = {k: v.clone() for k, v in pipeline.transformer.named_parameters()}
    
    # Zastosowanie wag LoRA do części 'transformer' w pipeline
    pipeline.transformer = apply_lora_to_model(pipeline.transformer, lora_state_dict, scaling_factor=1.0)

    # (Opcjonalnie) Sprawdzenie, czy wagi uległy zmianie
    for name, param in pipeline.transformer.named_parameters():
        if name in original_transformer_state:
            diff = (param.data - original_transformer_state[name]).abs().sum().item()
            if diff > 0:
                print(f"[DEBUG] {name} został zmodyfikowany: suma różnic = {diff:.4f}")

    # Generowanie obrazu – przykładowy prompt (upewnij się, że długość promptu mieści się w ograniczeniach modelu)
    prompt = "A high quality photorealistic image of a cat sitting on a metallic surface, adding a sense of action and tension."
    
    # Jeśli używasz CLIPTextModel, pamiętaj o ograniczeniu długości sekwencji (np. do 77 tokenów)
    outputs = pipeline(prompt)
    
    # Zakładamy, że pipeline zwraca listę obrazów – zapisujemy pierwszy z nich
    output_image = outputs[0]
    output_path = "/tmp/output_0.webp"
    output_image.save(output_path, "WEBP", quality=80)
    print(f"[INFO] Obraz zapisany do {output_path}")
