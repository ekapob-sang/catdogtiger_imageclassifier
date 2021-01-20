# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:47:42 2021

@author: admin-1309
"""

from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import  load_model

def catdogtiger_classifier(image):
    # Load the model
    model = tf.keras.models.load_model("catdog_model2_tf20.h5")
    # json_file = open('G:\My Drive\catdogtiger2\model_set2\model2.json', 'r')
    # model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(model_json)
    # model.load_weights('G:\My Drive\catdogtiger2\model_set2\model2_weights.h5')
    size = (64, 64)
    uploaded_file  = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = uploaded_file.convert('RGB')
    image_array= np.asarray(image)
    # Normalize the image
    normalized_image_array = image_array.astype(np.float32) / 255
    #reshape
    img_reshape = normalized_image_array[np.newaxis,...]
    # Load the image into the array
    #data[0] = normalized_image_array  # (Not sure if this is needed, but gives an error!!!)
    # run the inference
    prediction = model.predict(img_reshape)  
    return prediction
