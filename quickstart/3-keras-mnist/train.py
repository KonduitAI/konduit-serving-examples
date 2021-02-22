import tensorflow as tf

from keras.datasets import mnist


tensorflow_version = tf.__version__
print(tensorflow_version)

# Load data
train_data, test_data = mnist.load_data()
x_train, y_train = train_data
x_test, y_test = test_data

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0


def get_model():
    inputs = tf.keras.layers.Input(shape=(28, 28), name="input_layer")
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    x = tf.keras.layers.Dense(100, activation="relu")(x)
    x = tf.keras.layers.Dense(60, activation="relu")(x)
    x = tf.keras.layers.Dense(30, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train(epochs=8):
    model = get_model()
    model.fit(x_train, y_train, epochs=epochs)

    model.summary()

    print("\n\n---\n"
          "Inputs: {}".format(model.inputs))
    print("Outputs: {}\n---".format(model.outputs))

    return model


train(8).save("keras.h5", save_format="h5")

