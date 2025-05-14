import os
from tqdm import tqdm
from keras.preprocessing.image import load_img
from PIL import Image
import pandas as pd
import re

import os
import pandas as pd
from tqdm import tqdm

def parse_image_metadata(dataset_path):
    image_paths = []
    age_labels = []

    # Manually list files in the directory with tqdm
    for root, dirs, files in os.walk(dataset_path):  # Use os.walk to traverse subdirectories
        for filename in tqdm(files, desc="Processing images"):  # Wrap with tqdm
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Split filename at first underscore
                    age_part, _ = filename.split('_', 1)
                    age = int(age_part)

                    image_paths.append(os.path.join(root, filename))  # full path
                    age_labels.append(age)

                except Exception as e:
                    print(f"❌ Skipping {filename}: {e}")  # Print the error message
                    continue

    if len(image_paths) != len(age_labels):
        raise ValueError(f"Mismatch between images ({len(image_paths)}) and labels ({len(age_labels)})")

    df = pd.DataFrame({
        'image': image_paths,
        'age': age_labels
    })

    print(f"DataFrame created with {len(df)} entries.")  # Debugging line
    return df



def augment_image(img, angles=[20, 40, -20, -40]):
    """Rotate the image by given angles and return list of augmented images."""
    augmented_images = []
    for angle in angles:
        augmented = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        augmented_images.append((augmented, angle))
    return augmented_images

def resize_and_augment_and_save(df, output_folder, image_size=(128, 128), grayscale=True):
    os.makedirs(output_folder, exist_ok=True)
    print(df.head(5))
    for image_path, age in tqdm(list(zip(df['image'], df['age'])), total=len(df)):
        try:
            img = load_img(image_path, color_mode='grayscale' if grayscale else 'rgb')
            img = img.resize(image_size, Image.Resampling.LANCZOS)

            # Save resized original image
            base_filename = os.path.basename(image_path)
            save_path = os.path.join(output_folder, base_filename)
            img.save(save_path)

            # Save augmented rotated images
            augmented_imgs = augment_image(img)
            for aug_img, angle in augmented_imgs:
                # Save with angle in filename
                name, ext = os.path.splitext(base_filename)
                new_filename = f"{name}_rot{angle}{ext}"
                aug_save_path = os.path.join(output_folder, new_filename)
                aug_img.save(aug_save_path)

        except Exception as e:
            print(f"❌ Skipping {image_path}: {e}")

if __name__ == "__main__":
    dataset_path = "data/combined_faces"    # original dataset
    output_folder = "data/processed_faces"   # 📦 where resized+augmented images will go

    os.makedirs(output_folder, exist_ok=True)

    print("📦 Parsing image metadata...")
    df = parse_image_metadata(dataset_path)

    print("🛠 Resizing, augmenting and saving images...")
    resize_and_augment_and_save(df, output_folder)

    print("✅ Done! All images saved to:", output_folder)
