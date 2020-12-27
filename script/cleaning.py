import pandas as pd
from clean import CleanText
import os
import nltk


# Define PATH files
DATA_PATH = '/Users/Morgane/Desktop/5GMM/DefiIA/data'#'/Users/cecile/Documents/INSA/DefiIA/data/'    #TODO
#DATA_PATH = '/home/cecile/data'  # PATH si utilisation de l'instance (attention il faut commenter les os.makedirs...)
DATA_CLEANED_PATH = os.path.join(DATA_PATH, 'cleaned')
if not os.path.exists(DATA_CLEANED_PATH):
    os.makedirs(DATA_CLEANED_PATH)


# Reading files
train_df = pd.read_json(DATA_PATH+"/train.json")
train_df.set_index('Id', inplace=True)
train_df['description'] = [line.replace('\r', '') for line in train_df["description"].values]

test_df = pd.read_json(DATA_PATH+"/test.json")
test_df.set_index('Id', inplace=True)
test_df['description'] = [line.replace('\r', '') for line in test_df["description"].values]

train_label = pd.read_csv(DATA_PATH+"/train_label.csv")
train_label.set_index('Id', inplace=True)


# Cleaning process and save
ct = CleanText(stemming=True, lem=False)  # TODO
#try:
ct.clean_save(train_df, 'train', "description", "description_cleaned", DATA_CLEANED_PATH)
ct.clean_save(test_df, 'test', "description", "description_cleaned", DATA_CLEANED_PATH)
#except:
#    nltk.download('stopwords')
#    nltk.download('wordnet')
#    ct.clean_save(train_df, 'train', "description", "description_cleaned", DATA_CLEANED_PATH)
#    ct.clean_save(test_df, 'test', "description", "description_cleaned", DATA_CLEANED_PATH)
