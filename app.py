# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 21:54:23 2021

@author: admin-1309
"""

import streamlit as st
#import keras
from PIL import Image, ImageOps
import numpy as np
from catdogclassifier import catdogtiger_classifier
import pandas as pd



st.title("Dog / Cat / Tiger Classfication Model")
st.text("Use the simple AI for classification ")
st.text("This model has accuracy about 77%")
st.text("- กด browse file เพื่อทำนายภาพ" )
st.text("- ถ้าต้องการทำนายภาพซ้ำ คลิก X เพื่อลบรูปเดิมออก" )
result = st.empty()



uploaded_file = st.file_uploader("Upload your picture here ...", type=["jpg","png","jpeg"],accept_multiple_files=False)
if uploaded_file is None:
   st.text("Upload your picture")
else:
  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded picture', use_column_width=True)
  st.write("")
  predict = catdogtiger_classifier(image) # Name of the model from Teachablemachine
  if np.argmax(predict) == 0:
        result.subheader("It is a dog!")
  elif np.argmax(predict) == 1:
        result.subheader("It is a cat!")
  else:
        result.subheader("It is a tiger!")
  predict_prob=np.round(predict,2)*100
  a=pd.DataFrame(predict_prob,columns=('Dog','Cat','Tiger'),index=['Probability(%)'])
  st.dataframe(a)


