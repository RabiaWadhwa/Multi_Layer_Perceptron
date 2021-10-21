from src.utils.common_utils import read_config, get_log_path
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model, save_plot

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf


def training(config_path):
    config = read_config(config_path)
    print(config)

    validation_datasize = config["params"]["validation_datasize"]
    print(f"Validation Datasize {validation_datasize}")
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    no_classes, loss_fn, optimiser, metric = config["params"]['no_classes'], config["params"]['loss_function'], \
                                             config["params"]['optimizer'], config["params"]['metrics']
    print(f"#Classes-{no_classes} Loss function-{loss_fn} Optimizer-{optimiser} Metrics-{metric}")

    model_classifier = create_model(metric, optimiser, loss_fn, no_classes)

    # Writing Sample Images to tensorboard
    log_dir_path = get_log_path()

    # Unique name folder is created inside Log/fit folder,
    # Tensorflow event file resides in this folder
    file_writer = tf.summary.create_file_writer(logdir=log_dir_path)
    with file_writer.as_default():
        # -1 refers to #images
        # 1 refers to 1 channel - black & white image
        images = np.reshape(X_train[:20], (-1, 28, 28, 1))  # Reshaping first 20 images ,28X28
        tf.summary.image("20 handwritten images", images, max_outputs=25, step=0)

    # To Run tensorboard,first execute the file
    # python training.py
    # tensorboard --logdir artifacts/log/fit
    # This command provides a localhost link to open tensorboard in browser

    # Tensorboard Callback
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir_path)

    # Early stopping callback
    # If validation accuracy isn't improving after last patience=5 epochs,
    # it'll stop the training & restore best weights achieved until then
    # (with best validation accuracy)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    # Model Checkpoint Callback
    # If system crashes while training the model, then save the last BEST model ,hence modelfile has'.h5' extension
    model_name = "models/model_checkpoint.h5"
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_name, save_best_only=True)

    callback_list = [tensorboard_cb, early_stop_cb, model_checkpoint_cb]

    trained_model = model_classifier.fit(X_train, y_train, epochs=config["params"]["epochs"], validation_data=(X_valid, y_valid), callbacks=callback_list)

    # Save model
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_name = config["artifacts"]["model_name"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    save_model(model_classifier, model_name, model_dir_path) # model_classifier is the model

    # Save Plot
    track_performance = pd.DataFrame(trained_model.history)
    # Gives model loss, accuracy & validation loss & accuracy for each of the 30 epochs
    print(f"Model,Validation Loss/Accuracy {track_performance}")

    plot_dir = config["artifacts"]["plot_dir"]
    plot_name = config["artifacts"]["plot_name"]

    plot_dir_path = os.path.join(artifacts_dir, plot_dir)
    os.makedirs(plot_dir_path, exist_ok=True)
    save_plot(track_performance, plot_name, plot_dir_path)


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument , # arg name - config or c , default value - config.yaml
    parser.add_argument("--config", "-c", default="config.yaml")

    # Parse the argument
    args = parser.parse_args()

    # Print User-input argument 
    print("Config input by user ", args.config)

    # Calling training function
    training(args.config)

# To pass the argument at runtime,
# python training.py -c = config.yaml
# python training.py --config = config.yaml  --secret= secret.yaml
# python training.py config.yaml secret.yaml  # positional arguments

