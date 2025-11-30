# """
# Transfer Learning
# """
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.applications import InceptionResNetV2
import tensorflow as tf


# # Define a custom layer for loading that handles multiple inputs
class CustomScaleLayer(Layer):
    def __init__(self, scale, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            # Apply scaling to each input tensor
            return [inp * self.scale for inp in inputs]
        else:
            # Apply scaling to a single input tensor
            return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config

# # Load the model with custom_objects
# model1 = load_model(r"Transfer_Learning_InceptionResnetV2_Soyabean_classification.hdf5", custom_objects={'CustomScaleLayer': CustomScaleLayer})
# model2 = load_model(r"feature_extraction_InceptionResnetV2_Soyabean_classification.hdf5", custom_objects={'CustomScaleLayer': CustomScaleLayer})
# print("Model loaded successfully!")



# from tensorflow.keras.preprocessing import image
# import numpy as np

# # Class labels
# classes = ['Broken', 'Immature', 'Intact', 'Skin Damaged', 'Spotted']

# img_path = r"C:\Users\Hari\Downloads\archive\Skin-damaged soybeans\96.bmp"

# # ==========================
# # Preprocess image for model1 / conv_base
# # ==========================
# img = image.load_img(img_path, target_size=(299, 299))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = x / 255.0

# # ==========================
# # Model1 prediction (full InceptionResNetV2)
# # ==========================
# pred1 = model1.predict(x)
# pred_class1 = classes[np.argmax(pred1)]
# confidence1 = np.max(pred1) * 100
# print(f"🧠 Model1 Prediction: {pred_class1} ({confidence1:.2f}%)")

# # ==========================
# # Model2 prediction (feature-based)
# # ==========================
# # Extract features first
# conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# features = conv_base.predict(x)  # shape (1, 8, 8, 1536)

# pred2 = model2.predict(features)
# pred_class2 = classes[np.argmax(pred2)]
# confidence2 = np.max(pred2) * 100
# print(f"🧠 Model2 Prediction: {pred_class2} ({confidence2:.2f}%)")







"""
Caption generator
"""

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

# Load HDF5 model and tokenizer
model = load_model(r"caption_generation_soyabean_model.hdf5")
with open(r"soyabean_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Feature extractor
conv_base = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = load_img(img_path, target_size=(299,299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return conv_base.predict(x)

def generate_caption(model, tokenizer, photo_feature, max_len):
    in_text = "startseq"
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = np.pad(seq, (0, max_len-len(seq)), 'constant')
        yhat = model.predict([photo_feature, np.array([seq])], verbose=0)
        yhat_idx = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_idx)
        if word is None or word=="endseq":
            break
        in_text += " " + word
    return in_text.replace("startseq", "").replace("endseq", "").strip()

# Example prediction
img_path = r"C:\Users\Hari\Downloads\archive\Skin-damaged soybeans\96.bmp"
features = extract_features(img_path)
caption = generate_caption(model, tokenizer, features,11 )
print("📝 Predicted caption:", caption)

