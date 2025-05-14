import os
import tensorflow as tf
import pandas as pd  # Used for saving history
from src.models.cnn_age_model import build_cnn
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import ModelCheckpoint

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ✅ Set memory growth for the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth set.")
    except RuntimeError as e:
        print(e)


# 🔥 Parse function to load & preprocess images
def parse_image(filename, image_size=(128, 128), grayscale=True):
    # Extract age from the filename (before the first "_")
    age_part = tf.strings.split(tf.strings.split(filename, os.sep)[-1], '_')[0]
    age = tf.strings.to_number(age_part, out_type=tf.float32)

    image = tf.io.read_file(filename)
    channels = 1 if grayscale else 3
    image = tf.image.decode_image(image, channels=channels, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    return image, age



# 🛠️ Build the dataset
def build_dataset(folder_path, batch_size=32, image_size=(128, 128), grayscale=True, shuffle_buffer_size=1000):
    # List all files (we assume .jpg files for now)
    pattern = os.path.join(folder_path, "*.jpg")
    matched_files = tf.io.gfile.glob(pattern)
    if not matched_files:
        raise ValueError(f"No image files found in {folder_path} with pattern *.jpg")

    print(f"✅ Found {len(matched_files)} images.")

    dataset = tf.data.Dataset.list_files(pattern, shuffle=True)
    dataset = dataset.map(
        lambda x: parse_image(x, image_size=image_size, grayscale=grayscale),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def train_from_dataset(folder_path, save_path="outputs/models/best_model.h5",
                       history_save_path="outputs/history/training_history.csv",
                       dataset_size=240000, batch_size=32):
    dataset = build_dataset(folder_path, batch_size=batch_size)

    # Split dataset into train and validation
    val_batches = int(0.2 * (dataset_size // batch_size))  # 20% for validation
    val_dataset = dataset.take(val_batches)
    train_dataset = dataset.skip(val_batches)

    model = build_cnn()

    model.compile(
        loss='mae', 
        optimizer='adam', 
        metrics=[MeanAbsoluteError()]
    )



    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=[checkpoint],
        verbose=1
    )

    # Save training history
    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_save_path, index=False)
    print(f"📈 Training history saved to: {history_save_path}")

    return model, history


if __name__ == "__main__":
    processed_folder = "data/processed_faces"
    output_model_path = "outputs/models/new_best_model.h5"
    output_history_path = "outputs/history/new_training_history.csv"

    model, history = train_from_dataset(
        processed_folder,
        output_model_path,
        output_history_path,
        dataset_size=240000,  # ⚠️ Adjust if needed
        batch_size=32
    )

    print("✅ Training complete.")
    print(f"✅ Model saved to: {output_model_path}")
    print(f"✅ History saved to: {output_history_path}")