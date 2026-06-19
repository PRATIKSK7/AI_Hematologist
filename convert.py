import tensorflow as tf
model = tf.keras.models.load_model("models/morphology_model.keras")
model.export("backend/models/morphology_savedmodel")
print("Exported successfully!")
