import tensorflow as tf


# Load Mnist Datasets
# Return Training,Validation & Test dataset
def get_data(validation_datasize):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    print(f"X-Train - ", {X_train_full.shape}, " X-Test- ",
          {X_test.shape})  # contains 60k,10k (train,test) black & white images of 28X28 size
    print(f"y-Train - ", {y_train_full.shape}, " y-Test- ",
          {y_test.shape})  # contains 60k,10k (train,test) labels for 10 distinct classes as it has images of 0-9 digits

    # First 5000(validation datasize) for validation,rest for training
    X_valid, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    # Dividing by 255 to rescale values between 0-1
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    X_test = X_test / 255
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
