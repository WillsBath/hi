# ==============================================================
# 1. MOUNT GOOGLE DRIVE
# ==============================================================
from google.colab import drive
drive.mount('/content/drive')

# ==============================================================
# 2. EXTRACT OR COPY DATASET FROM DRIVE
# ==============================================================

# If your dataset is a ZIP file (example: Alzheimer.zip)
# Uncomment this line:
# !unzip /content/drive/MyDrive/Alzheimer.zip -d /content/

# If your dataset is ALREADY a folder in Drive:
# !cp -r /content/drive/MyDrive/Alzheimer /content/

# After this, your test directory becomes:
# /content/Alzheimer/test


# ==============================================================
# 3. IMPORTS
# !pip install numpy pickle5 tensorflow pillow scikit-learn matplotlib
# ==============================================================

import numpy as np
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# --------------------------
# Load tokenizer
# --------------------------
tokenizer = load(open('/content/drive/MyDrive/tokenizer_inceptionv3_alzheimer.pkl', 'rb'))
vocab_size = len(tokenizer.word_index) + 1
max_length = 34  # Keep same as during training

# --------------------------
# Recreate captioning model
# --------------------------
def recreate_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))  # InceptionV3 features
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, use_cudnn=False)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# --------------------------
# Load model and weights
# --------------------------
model = recreate_model(vocab_size, max_length)
model.load_weights('/content/drive/My Drive/Caption_InceptionV3.h5')
print("Model loaded successfully!")

# --------------------------
# Feature extraction with InceptionV3
# --------------------------
def extract_feature(filename):
    model_incep = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    image = load_img(filename, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model_incep.predict(image, verbose=0)
    return feature

# --------------------------
# Word ID mapping
# --------------------------
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# --------------------------
# Generate caption
# --------------------------
def generate_caption(model, tokenizer, photo_feature, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()[1:-1]  # remove startseq and endseq
    final_caption = ' '.join(final_caption)
    return final_caption

# --------------------------
# Test prediction
# --------------------------
test_image_path = '/content/drive/My Drive/Alzheimer/test/nonDem87.jpg'
photo_feature = extract_feature(test_image_path)
caption = generate_caption(model, tokenizer, photo_feature, max_length)
print("Predicted Caption:", caption)
