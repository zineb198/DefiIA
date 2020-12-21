import pandas as pd
from clean import CleanText
import os
import nltk


# Define PATH files
DATA_PATH = '/Users/cecile/Documents/INSA/DefiIA/data/'    #TODO
DATA_CLEANED_PATH = os.path.join(DATA_PATH, 'cleaned')
if not os.path.exists(DATA_CLEANED_PATH):
    os.makedirs(DATA_CLEANED_PATH)

nanlist = [98883, 154202, 179598, 213351]   # A modif dans une fonction clean.py


# Reading files
train_df = pd.read_json(DATA_PATH+"/train.json")
train_df.set_index('Id', inplace=True)

test_df = pd.read_json(DATA_PATH+"/test.json")
test_df.set_index('Id', inplace=True)

train_label = pd.read_csv(DATA_PATH+"/train_label.csv")
train_label.set_index('Id', inplace=True)

categ = pd.read_csv(DATA_PATH+'/categories_string.csv')

train_df.drop(nanlist, axis=0, inplace=True)
train_label.drop(nanlist, axis=0, inplace=True)


# Cleaning process and save
ct = CleanText(lem=True, stemming=False)
try:
    ct.clean_save(train_df, 'train', "description", "description_cleaned", DATA_CLEANED_PATH)
    ct.clean_save(test_df, 'test', "description", "description_cleaned", DATA_CLEANED_PATH)
    train_label['Category'].to_csv(os.path.join(DATA_CLEANED_PATH,'train_label.csv'), index=True)
except:
    nltk.download('wordnet')
    ct.clean_save(train_df, "description", "description_cleaned", DATA_CLEANED_PATH)
    ct.clean_save(test_df, "description", "description_cleaned", DATA_CLEANED_PATH)
    train_label['Category'].to_csv(os.path.join(DATA_CLEANED_PATH,'train_label.csv'), index=True)