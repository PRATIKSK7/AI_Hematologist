import tensorflow as tf

class PatchedRescaling(tf.keras.layers.Rescaling):
    def __init__(self, scale, offset=0.0, **kwargs):
        super().__init__(scale=scale, offset=offset)

class PatchedNormalization(tf.keras.layers.Normalization):
    def __init__(self, axis=-1, mean=None, variance=None, invert=False, **kwargs):
        super().__init__(axis=axis, mean=mean, variance=variance, invert=invert)

custom_objects = {
    'Rescaling': PatchedRescaling,
    'Normalization': PatchedNormalization
}

try:
    model = tf.keras.models.load_model('models/morphology_model.keras', custom_objects=custom_objects)
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
