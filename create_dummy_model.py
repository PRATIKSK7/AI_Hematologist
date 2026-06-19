import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import os

print("Creating dummy model...")
input_layer = Input(shape=(224, 224, 3))
base_model = EfficientNetB0(include_top=False, weights=None, input_tensor=input_layer)
x = GlobalAveragePooling2D()(base_model.output)
output_layer = Dense(11, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)
os.makedirs("backend/models", exist_ok=True)
model.save("backend/models/morphology_model.keras")
print("Dummy model saved.")
