from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.applications import VGG16
from keras.optimizers import Adam
import keras


def build_model():

    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    output = base_model.layers[-1].output

    output = keras.layers.Flatten()(output)

    base_model = Model(
        base_model.input,
        output
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()

    model.add(base_model)

    model.add(Dense(
        512,
        activation='relu'
    ))

    model.add(Dropout(0.3))

    model.add(Dense(
        512,
        activation='relu'
    ))

    model.add(Dropout(0.3))

    model.add(Dense(
        1,
        activation='sigmoid'
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(
            learning_rate=0.0001
        ),
        metrics=['accuracy']
    )

    return model


def train_model(
        model,
        train_generator,
        valid_generator,
        epochs=5
):

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs
    )

    return history


def evaluate_model(
        model,
        test_generator
):

    return model.evaluate(
        test_generator,
        steps=test_generator.samples //
        test_generator.batch_size
    )
