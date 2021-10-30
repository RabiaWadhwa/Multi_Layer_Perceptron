import tensorflow as tf
import os
import time
import numpy as np


# Creating Tensorboard Log folder with unique name
def get_log_path(log_dir):
    dirname = time.strftime("LOG_%Y_%M_%d__%H_%M_%S")
    dirpath = os.path.join(log_dir, dirname)
    print(f"Saving logs at : {dirpath}")
    return dirpath


def get_callbacks(config,X_train):
    tensorboard_log_dir_path = os.path.join(config["logs"]["log_dir"],config["logs"]["tensorboard_logs"])
    os.makedirs(tensorboard_log_dir_path,exist_ok=True)
    log_dir_path = get_log_path(tensorboard_log_dir_path)

    # Writing Sample Images to tensorboard
    
    # Unique name folder is created inside log_dir/tensorboard_logs,
    # Tensorflow event file resides in this folder
    file_writer = tf.summary.create_file_writer(logdir=log_dir_path)
    with file_writer.as_default():
        # -1 refers to #images
        # 1 refers to 1 channel - black & white image
        images = np.reshape(X_train[:20], (-1, 28, 28, 1))  # Reshaping first 20 images ,28X28
        tf.summary.image("20 handwritten images", images, max_outputs=25, step=0)

    # To Run tensorboard,first execute the file
    # python training.py
    # Then,run following command at bash terminal
    # tensorboard --logdir log_dir/tensorboard_logs
    # This command provides a localhost link to open tensorboard in browser

    # Tensorboard Callback
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir_path)

    # Early stopping callback
    # If validation accuracy isn't improving after last patience=2 epochs,
    # it'll stop the training & restore best weights achieved until then
    # (with best validation accuracy)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(patience=config["params"]["patience"], restore_best_weights=config["params"]["restore_best_weights"])

    # Model Checkpoint Callback
    # If system crashes while training the model, then save the last BEST model ,hence modelfile has'.h5' extension
    checkpoint_dir = os.path.join(config["artifacts"]["artifacts_dir"],config["artifacts"]["checkpoint_dir"])
    os.makedirs(checkpoint_dir,exist_ok=True)
    model_checkpoint_name = "model_chkpoint.h5"
    model_checkpoint_path = os.path.join(checkpoint_dir,model_checkpoint_name)
    print(f"Writing model checkpoints at {model_checkpoint_path}")
    
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, save_best_only=config["params"]["save_best_only"])

    callback_list = [tensorboard_cb, early_stop_cb, model_checkpoint_cb]
    return callback_list

    

    