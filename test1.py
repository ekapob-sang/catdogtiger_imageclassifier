# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 08:20:45 2021

@author: admin-1309
"""

import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
from keras.models import model_from_json

# model_path ="E:\Practice deploy\catdogtiger_classifier\model.h5"
# model = tf.keras.models.load_model(model_path)

json_file = open('G:\My Drive\catdogtiger2\model_set2\model2.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('G:\My Drive\catdogtiger2\model_set2\model2_weights.h5')




st.title("Dog / Cat / Tiger Classfication ")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"],accept_multiple_files=False)

def import_and_predict(image_data, model): 
        size = (64,64)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(64, 64),    interpolation=cv2.INTER_CUBIC))/255.
        img_reshape = img_resize[np.newaxis,...]
        prediction = model.predict(img_reshape)  
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model) 
    if np.argmax(prediction) == 0:
        st.write("It is a dog!")
    elif np.argmax(prediction) == 1:
        st.write("It is a cat!")
    else:
        st.write("It is a tiger!")
    
    st.text("Probability (%) (0: Dog, 1: Cat, 2: Tiger")
    st.write(np.round(prediction*100,2))