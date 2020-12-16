import pandas as pd
from clean import CleanText
import os
import nltk


DATA_PATH = "/home/bennis/Bureau/5GMM/AI-Frameworks/DefiIA/data"     #TODO

nanlist = [98883, 154202, 179598, 213351]

# Reading files
train_df = pd.read_json(DATA_PATH+"/train.json")
train_df.set_index('Id', inplace=True)

test_df = pd.read_json(DATA_PATH+"/test.json")
test_df.set_index('Id', inplace=True)

train_label = pd.read_csv(DATA_PATH+"/train_label.csv")
train_label.set_index('Id', inplace=True)

categ = pd.read_csv(DATA_PATH+'/categories_string.csv')

train_df.drop(nanlist, axis = 0, inplace = True)
train_label.drop(nanlist, axis = 0, inplace = True)


# Cleaning process
ct = CleanText(lem=True, stemming=False)   #args avec commande python
try :
    ct.clean_df_column(train_df, "description", "description_cleaned")
    ct.clean_df_column(test_df, "description", "description_cleaned")
except : #A mettre dans requierements
    nltk.download('wordnet')
    ct.clean_df_column(train_df, "description", "description_cleaned")
    ct.clean_df_column(test_df, "description", "description_cleaned")


''''''

# Save df cleaned
DATA_CLEANED_PATH = DATA_PATH+'/cleaned/'
if not os.path.exists(DATA_CLEANED_PATH):
    os.makedirs(DATA_CLEANED_PATH)

# Changer noms de fichiers et faire que train_label n'aie qu'une seule colonne (id en trop)
train_df.to_csv(DATA_CLEANED_PATH+'cleaned_train.csv', index=True)
test_df.to_csv(DATA_CLEANED_PATH+'cleaned_test.csv', index=True)
train_label['Category'].to_csv(DATA_CLEANED_PATH+'train_label.csv', index=True)