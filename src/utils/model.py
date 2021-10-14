import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt


def create_model(metric, optimiser, loss_fn, no_classes):
    LAYERS = [
        # Flatten (28X28) each Image's data points into single 1-D array - 764 inputs
        tf.keras.layers.Flatten(input_shape=[28, 28], name="Input_Layer"),
        tf.keras.layers.Dense(300, activation="relu", name="Hidden_Layer1"),
        tf.keras.layers.Dense(100, activation="relu", name="Hidden_Layer2"),
        tf.keras.layers.Dense(no_classes, activation="softmax", name="Output_Layer")
    ]

    model_classifier = tf.keras.models.Sequential(LAYERS)
    # sequential model, no skip connection between any of the layers

    print(model_classifier.summary())
    model_classifier.compile(metrics=metric, loss=loss_fn, optimizer=optimiser)

    return model_classifier


def create_unique_file_name(filename):
    unique_file_name = time.strftime(f"%Y_%m_%d__%H_%M_%S_{filename}.h5")  # E.g. '2021_10_13__23_47_35_model.h5'
    # no spaces or special characters allowed in filename
    return unique_file_name


def save_model(model, filename, model_dir):
    model_name = create_unique_file_name(filename)
    file_path = os.path.join(model_dir, model_name) # model/filename
    print(f"Model path - {file_path}")
    model.save(file_path)  # saves New model version with new timestamp ,everytime program is run


def save_plot(df, plot_name, plot_dir):
    df.plot(figsize=(10,7))
    plt.grid(True)

    plot_path = os.path.join(plot_dir, plot_name)  # plot/plot_name
    print(f"Plot path - {plot_path}")
    plt.savefig(plot_path)
    # Don't use plt.show() before savefig()

