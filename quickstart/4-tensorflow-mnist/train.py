import tensorflow as tf

if tf.__version__[0] == "1":     
    from tensorflow import keras
elif tf.__version__[0] == "2": 
    import tensorflow.compat.v1 as tf
    from tensorflow.compat.v1 import keras
else: 
    print("No valid TensorFlow version detected")

from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.models import Sequential
from keras.datasets import mnist

import numpy as np
import os
import pandas as pd

tensorflow_version = tf.__version__
print(tensorflow_version)

# Load data
train_data, test_data = mnist.load_data()
x_train, y_train = train_data
x_test, y_test = test_data

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0


def get_model(training=False):
    inputs = keras.layers.Input(shape=(28, 28), name="input_layer")
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(200, activation="relu")(x)
    x = keras.layers.Dense(100, activation="relu")(x)
    x = keras.layers.Dense(60, activation="relu")(x)
    x = keras.layers.Dense(30, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax", name="output_layer")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    if training:
        print(model.inputs[0].op.name)
        print(model.outputs[0].op.name)

    return model


def train():
    with tf.Session() as sess:
        keras.backend.set_session(sess)
        model = get_model(True)
        model.fit(x_train, y_train, epochs=8)
        weights = model.get_weights()
    return weights


def save(weights):
    # save model to a protobuf
    keras.backend.clear_session()
    with tf.Session() as sess:
        keras.backend.set_session(sess)
        model = get_model(False)
        model.set_weights(weights)
        model.evaluate(x_test, y_test)
        output_node_name = model.output.name.split(':')[0]
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            [output_node_name]
        )

        with tf.gfile.GFile(
            name=f"tensorflow.pb",
            mode="wb"
        ) as f:
            f.write(output_graph_def.SerializeToString())


save(train())
