import pandas as pd
from sklearn.model_selection import train_test_split

"""
Script to create test set that we will not touch during our project.
We will stratify by label and day_phase
"""

df = pd.read_pickle('data/raw/allAnnotations20210422_pkv4.pkl')
train, test = train_test_split(df, test_size=0.2, stratify=df[['label', 'day_phase']], random_state=0)
train.to_pickle('data/raw/annotations_train.pkl')
test.to_pickle('data/raw/annotations_test.pkl')
