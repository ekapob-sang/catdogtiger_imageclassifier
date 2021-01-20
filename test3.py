# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 21:54:23 2021

@author: admin-1309
"""

import streamlit as st
#import keras
from PIL import Image, ImageOps
import numpy as np
from test3_classifier import catdogtiger_classifier


st.title("Dog / Cat / Tiger Classfication ")

st.text("Upload your picture here")
uploaded_file = st.file_uploader("Upload your picture here ...", type=["jpg","png"],accept_multiple_files=False)
if uploaded_file is None:
   st.text("Upload your picture")
else:
  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded picture', use_column_width=True)
  st.write("")
  predict = catdogtiger_classifier(image) # Name of the model from Teachablemachine
  if np.argmax(predict) == 0:
        st.write("It is a dog!")
  elif np.argmax(predict) == 1:
        st.write("It is a cat!")
  else:
        st.write("It is a tiger!")
  st.text("Probability (%) 0: Dog, 1: Cat, 2: Tiger")
  st.write(np.round(predict*100,2))