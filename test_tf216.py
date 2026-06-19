import tensorflow as tf
print("TF Version:", tf.__version__)
model = tf.keras.models.load_model("models/morphology_model.keras")
model.summary()
print("Success!")
