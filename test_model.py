import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
model = tf.keras.models.load_model("backend/models/morphology_savedmodel")

# In TF 2.15, SavedModels sometimes load as an object instead of a Keras model if exported from Keras 3.
# Let's check type.
print(type(model))

try:
    model.summary()
except Exception as e:
    print("Cannot print summary:", e)
    
# Test an inference to be sure
import numpy as np
dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
pred = model(dummy_input)
print("Output shape:", pred.shape)
