import pandas as pd
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

os.chdir('/home/ubuntu/Thesis-KNMI-FoggyGAN/')
df = pd.read_pickle('data/raw/train_annotations.pkl')

# Discard daytime images
keep = [0, 20, 21, 30, 31]
df = df[df.day_phase.isin(keep)]

# Discard cannot say images
df = df[df.label != 'Cannot Say']

df_translations = pd.read_pickle('data/processed/annotations_generation1.pkl') # PATH TO ANNOTATIONS OF TRANSLATIONS
df = pd.concat([df, df_translations])

df_foggy = df[df.label == 'Fog']
df_clear = df[df.label == 'No Fog'].sample(len(df_foggy), random_state=17) 

df_train = pd.concat([df_foggy, df_clear])

# MAP STRING LABELS TO INTEGERS
mapping = {'No Fog':0, 'Fog':1}
y = df_train['label'].map(mapping)


# LOAD IMAGES
size = (256, 256)

images = []

# If filename starts with translated, look in folder where translations are. If not, look in folder with all training images.
for fname in df_train.filename:
    if fname.startswith('TRANSLATED'):
        datadir = 'data/processed/generated_images1'
    else:
        datadir = 'data/raw/train'
    path = os.path.join(datadir, fname)
    img = load_img(path, target_size=size)
    img = img_to_array(img)
    img=img/255
    images.append(img)
    
images = np.asarray(images)


#TRAIN/VALIDATION SPLIT
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.3, random_state=17)

# DEFINE MODEL
model = Sequential()
model.add(Conv2D(12, kernel_size=10, activation='relu', input_shape=(256, 256, 3)))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(8, kernel_size=5, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(8, kernel_size=5, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(4, kernel_size=5, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# TRAIN MODEL
callbacks = EarlyStopping(patience=20, monitor='val_accuracy', min_delta=0, mode='max')

history = model.fit(X_train, y_train, 
                    batch_size=32, 
                    epochs=50, 
                    verbose=1, 
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    shuffle=True)

model.save('models/new/model1.h5') # SAVE MODEL