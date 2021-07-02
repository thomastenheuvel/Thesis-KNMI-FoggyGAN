import os
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

os.chdir('/home/ubuntu/Thesis-KNMI-FoggyGAN/')

def load_images(path, size=(384, 480)):
    data_list = list()
    for filename in os.listdir(path):
        # Load and resize image:
        pixels = load_img(os.path.join(path, filename), target_size=size)
        pixels = img_to_array(pixels) # Convert to numpy array
        data_list.append(pixels) # Store
    return asarray(data_list)
    
data_A = load_images('data/raw/trainA', size=(256, 256))
data_B = load_images('data/raw/trainB', size=(256, 256))

# Save dataset as compressed numpy array
fpath = 'data/processed/clear2foggy256x256.npz'
savez_compressed(fpath, data_A, data_B)