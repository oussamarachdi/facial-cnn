from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ReLU

def build_cnn(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)

    # Conv block 1
    x = Conv2D(32, (3, 3), activation=None)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv block 2
    x = Conv2D(64, (3, 3), activation=None)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv block 3
    x = Conv2D(128, (3, 3), activation=None)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv block 4
    x = Conv2D(256, (3, 3), activation=None)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    # Age prediction branch
    age_branch = Dense(256, activation='relu')(x)
    age_branch = Dropout(0.4)(age_branch)
    age_output = Dense(1, activation='linear')(age_branch)
    model = Model(inputs=inputs, outputs=age_output)


    return model
