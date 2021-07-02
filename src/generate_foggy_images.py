import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from numpy import expand_dims
from keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
from keras.preprocessing.image import save_img

os.chdir('/home/ubuntu/Thesis-KNMI-FoggyGAN')
df = pd.read_pickle('data/raw/train_annotations.pkl')

# Discard daytime images
keep = [0, 20, 21, 30, 31]
df = df[df.day_phase.isin(keep)]

# Discard cannot say images
df = df[df.label != 'Cannot Say']

# Discard foggy images
df = df[df.label != 'Fog']

# Sample 200 non-foggy images:
df = df.sample(200, random_state=17)

# LOAD IMAGES
def load_image(filename, size=(256,256)):
    # load and resize the image
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # transform in a sample
    pixels = expand_dims(pixels, 0)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    return pixels

datafolder = 'data/raw/images_data'
images = []

for fname in df.filename.values:
    path = os.path.join(datafolder, fname)
    image = load_image(path)
    images.append(image)
    
# LOAD MODEL
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('PATH/TO/GAN/MODEL.h5', cust) # REPLACE PATH TO MODEL

# TRANSLATE IMAGES
translations = []

for image in images:
    translation = model_AtoB.predict(image)
    translation = (translation+1)/2 # Rescale back from [-1, 1] to [0,1]
    translations.append(translation)
    
# SAVE TRANSLATIONS
target_dir = 'PATH/TO/TARGET/DIR'
source_fnames = df.filename.values

for i, translation in enumerate(translations):
    target_fname = 'TRANSLATED_'+source_fnames[i]
    target_path = os.path.join(target_dir, target_fname)
    save_img(target_path, translation[0],
             data_format='channels_last',
             scale=True)
    
# SAVE ANNOTATIONS
fnames_translations = ['TRANSLATED_' + fname for fname in df.filename.values]
df_translations = df.copy()
df_translations['filename'] = fnames_translations
df_translations['label'] = ['Fog'] * len(fnames_translations)
df_translations.to_pickle('TARGET/FILEPATH/ANNOTATIONS.pkl')
