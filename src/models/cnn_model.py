from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ReLU

def build_cnn(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation=None)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation=None)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation=None)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation=None)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    gender_branch = Dense(256, activation='relu')(x)
    gender_branch = Dropout(0.4)(gender_branch)
    gender_output = Dense(1, activation='sigmoid', name='gender_out')(gender_branch)

    age_branch = Dense(256, activation='relu')(x)
    age_branch = Dropout(0.4)(age_branch)
    age_output = Dense(1, activation='relu', name='age_out')(age_branch)

    model = Model(inputs=inputs, outputs=[gender_output, age_output])

    return model
