import matplotlib.pyplot as plt
import numpy as np

gender_dict = {0: 'Male', 1: 'Female'}

def plot_training_history(history):
    acc = history.history['gender_out_accuracy']
    val_acc = history.history['val_gender_out_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    age_mae = history.history['age_out_mae']
    val_age_mae = history.history['val_age_out_mae']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Train Acc')
    plt.plot(epochs, val_acc, 'r', label='Val Acc')
    plt.title('Gender Accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Val Loss')
    plt.title('Overall Loss')
    plt.legend()
    plt.figure()

    plt.plot(epochs, age_mae, 'b', label='Train MAE (Age)')
    plt.plot(epochs, val_age_mae, 'r', label='Val MAE (Age)')
    plt.title('Age MAE')
    plt.legend()
    plt.show()

def show_prediction(model, X, y_gender, y_age, idx):
    pred = model.predict(X[idx].reshape(1, 128, 128, 1))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    actual_gender = gender_dict[y_gender[idx]]
    actual_age = y_age[idx]

    print(f"Index: {idx}")
    print(f"Actual: {actual_gender}, {actual_age}")
    print(f"Predicted: {pred_gender}, {pred_age}")

    plt.axis('off')
    plt.imshow(X[idx].reshape(128, 128), cmap='gray')
    plt.show()
