import os
import torch
import clip
from PIL import Image

def get_user_input_directory():
    return input("Enter the directory path: ")

def process_file_name(file_name):
    image_path = os.path.join(directory, file_name)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(f"Label probabilities for '{file_name}': {probs}")

def list_files_in_directory(directory):
    try:
        file_list = os.listdir(directory)
        print("Files in the directory:")
        for file_name in file_list:
            print(file_name)
            process_file_name(file_name)
    except FileNotFoundError:
        print("Directory not found.")
    except PermissionError:
        print("Permission denied.")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    user_choice = input("Do you want to use the present working directory? (y/n): ")
    if user_choice.lower() == 'y':
        current_directory = os.getcwd()
        list_files_in_directory(current_directory)
    elif user_choice.lower() == 'n':
        user_input_directory = get_user_input_directory()
        list_files_in_directory(user_input_directory)
    else:
        print("Invalid choice. Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()
