import tensorflow as tf

def create_model(metric,optimiser,loss_fn,no_classes):
    LAYERS = [
                # Flatten (28X28) each Image's data points into single 1-D array - 764 inputs
                tf.keras.layers.Flatten(input_shape =[28,28] , name="Input_Layer"), 
                tf.keras.layers.Dense(300,activation="relu",name = "Hidden_Layer1"),
                tf.keras.layers.Dense(100,activation="relu",name = "Hidden_Layer2"),
                tf.keras.layers.Dense(no_classes, activation="softmax",name = "Output_Layer")
            ]

    model_classfier = tf.keras.models.Sequential(LAYERS) # sequential model, no skip connection between any of the layers
   
    print(model_classfier.summary())
    model_classfier.compile(metrics=metric ,loss=loss_fn ,optimizer=optimiser)

    return model_classfier

    