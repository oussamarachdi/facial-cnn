import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_age_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['age'], kde=True, bins=30, color='salmon')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_gender_distribution(df: pd.DataFrame):
    gender_map = {0: 'Male', 1: 'Female'}
    counts = df['gender'].map(gender_map).value_counts()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, palette='pastel')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def show_sample_images(df: pd.DataFrame, num_images=10):
    from keras.preprocessing.image import load_img
    import numpy as np

    gender_dict = {0: 'Male', 1: 'Female'}
    plt.figure(figsize=(12, 6))

    for i in range(num_images):
        row = df.iloc[i]
        img = load_img(row['image'])
        img = np.array(img)

        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f"Age: {row['age']} - {gender_dict[row['gender']]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
