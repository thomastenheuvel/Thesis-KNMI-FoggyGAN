import os
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

os.chdir('/home/ubuntu/Thesis-KNMI-FoggyGAN/')
df_test = pd.read_pickle('data/raw/test_annotations.pkl')

keep = [0, 20, 21, 30, 31]
df_test = df_test[df_test.day_phase.isin(keep)] # Drop day images
df_test = df_test[df_test.label != 'Cannot Say'] # Drop images with label cannot say

# MAP LABELS TO INTEGERS
mapping = {'No Fog':0, 'Fog':1}
y_true = df_test['label'].map(mapping)

# LOAD MODEL
model = load_model('models/new/model1.h5')

# PREPROCESSING
datadir = 'data/raw/test'
images = []


for fname in df_test.filename.values:
    img = load_img(os.path.join(datadir, fname), target_size=(256, 256))
    img = img_to_array(img)
    img = img/255
    images.append(img)

images = np.asarray(images)

# PREDICT
probas = model.predict(images)
y_pred = (probas > 0.5).astype('int32')

# EVALUATE
labels = ['No Fog', 'Fog']

cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot()

cr = classification_report(y_true, y_pred, target_names=labels)
print(cr)