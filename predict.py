import io
from PIL import Image

def apply_mask(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Łączy oryginalny obraz z wygenerowanym fragmentem na podstawie maski.
    Tam, gdzie maska ma wartość 255, wstawiany jest fragment z wygenerowanego obrazu.
    """
    # Dopasowanie rozmiarów obrazów
    if original.size != generated.size:
        generated = generated.resize(original.size)
    if original.size != mask.size:
        mask = mask.resize(original.size)
    
    # Konwersja maski do trybu szarości i binaryzacja (próg 128)
    mask = mask.convert("L")
    threshold = 128
    mask = mask.point(lambda p: 255 if p > threshold else 0)
    
    # Łączenie obrazów: gdzie maska = 255, wybierany jest obraz wygenerowany, w przeciwnym razie oryginalny
    final_image = Image.composite(generated, original, mask)
    return final_image

def predict(
    prompt: str,
    extra_lora: str,
    image,  # Obiekt PIL.Image (przekonwertowany przez Cog)
    mask,   # Obiekt PIL.Image (przekonwertowany przez Cog)
    output_format: str = "png",
) -> Image.Image:
    """
    Funkcja predict przyjmuje prompt, ścieżkę do wagi LoRA, obraz wejściowy oraz maskę.
    Generuje nowy obraz, nakładając wygenerowany fragment tylko w obszarach określonych maską.
    """
    # Konwersja obrazu wejściowego do formatu RGB
    input_image = image.convert("RGB")
    # Konwersja maski do skali szarości
    input_mask = mask.convert("L")
    
    # !!! TU WPISZ LOGIKĘ SWOJEGO MODELU !!!
    # Poniżej znajduje się placeholder – dla demonstracji używamy kopii obrazu jako wygenerowanego obrazu.
    # W rzeczywistości powinieneś wywołać funkcję, która generuje obraz na podstawie prompta oraz extra_lora.
    generated_image = input_image.copy()
    
    # Nałożenie maski: fragment wygenerowany zastępuje oryginał w obszarach wskazanych maską
    final_image = apply_mask(input_image, generated_image, input_mask)
    
    return final_image
