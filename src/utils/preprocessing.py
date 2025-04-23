import os
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import load_img
from PIL import Image
import pandas as pd

def parse_image_metadata(dataset_path):
    image_paths = []
    age_labels = []
    gender_labels = []

    for filename in tqdm(os.listdir(dataset_path)):
        try:
            age, gender, *_ = filename.split('_')
            image_path = os.path.join(dataset_path, filename)
            image_paths.append(image_path)
            age_labels.append(int(age))
            gender_labels.append(int(gender))
        except ValueError:
            continue

    df = pd.DataFrame({
        'image': image_paths,
        'age': age_labels,
        'gender': gender_labels
    })

    return df

def extract_features(images, image_size=(128, 128), grayscale=True):
    features = []
    for image_path in tqdm(images):
        img = load_img(image_path, color_mode='grayscale' if grayscale else 'rgb')
        img = img.resize(image_size, Image.Resampling.LANCZOS)
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    if grayscale:
        features = features.reshape(len(features), *image_size, 1)
    return features / 255.0  # Normalize

# 🔽 Entry point to run directly
if __name__ == "__main__":
    dataset_path = "data/UTKFace"  # Adjust if needed
    output_dir = "outputs/preprocessed"

    os.makedirs(output_dir, exist_ok=True)

    print("📦 Parsing image metadata...")
    df = parse_image_metadata(dataset_path)

    print("🖼️ Extracting features...")
    X = extract_features(df['image'])
    y_gender = np.array(df['gender'])
    y_age = np.array(df['age'])

    print("💾 Saving preprocessed arrays...")
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y_gender.npy"), y_gender)
    np.save(os.path.join(output_dir, "y_age.npy"), y_age)

    print("✅ Done! Features saved to:", output_dir)
