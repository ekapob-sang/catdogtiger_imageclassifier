# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 08:20:45 2021

@author: admin-1309
"""

import tensorflow as tf
import streamlit as st
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
