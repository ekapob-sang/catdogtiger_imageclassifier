# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 08:20:45 2021

@author: admin-1309
"""

import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import h5py


model = load_model('/app/model.h5')

st.title("Dog / Cat / Tiger Classfication ")
