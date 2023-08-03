import sys
import os
import torch
import clip
from PIL import Image

def process_file_name(file_name):
    image_path = os.path.join(directory, file_name)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(f"Label probabilities for '{file_name}': {probs}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    try:
        for file_name in sys.stdin.readlines():
            file_name = file_name.strip()
            if not file_name:
                continue
            print(file_name)
            process_file_name(file_name)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
