import os
import pickle
import numpy as np
from src.models.cnn_model import build_cnn
from src.utils.preprocessing import parse_image_metadata, extract_features
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Accuracy, MeanAbsoluteError

from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def train(dataset_path, save_path="outputs/models/best_model.h5"):
    df = parse_image_metadata(dataset_path)

    X = extract_features(df['image'])
    y_gender = np.array(df['gender'])
    y_age = np.array(df['age'])

    X_train, X_val, y_gender_train, y_gender_val, y_age_train, y_age_val = train_test_split(
        X, y_gender, y_age, test_size=0.2, random_state=42
    )

    model = build_cnn()
    

    model.compile(
        loss=['binary_crossentropy', MeanAbsoluteError()],
        optimizer='adam',
        metrics=[Accuracy(), MeanAbsoluteError()]
    )

    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        x=X_train,
        y=[y_gender_train, y_age_train],
        validation_data=(X_val, [y_gender_val, y_age_val]),
        epochs=30,
        batch_size=32,
        callbacks=[checkpoint]
    )

    return model, history


if __name__ == "__main__":
    dataset_path = "data/UTKFace"
    output_path = "outputs/models/best_model.h5"
    model, history = train(dataset_path, output_path)
    print("✅ Training complete.")
