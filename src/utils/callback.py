import tensorflow as tf
from src.utils.common_utils import get_log_path

def get_callbacks(config,X_train):
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

    