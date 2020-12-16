import re
import unicodedata
import nltk
import numpy as np
from tqdm import tqdm


class CleanText:
    '''
    This class is used to clean df with text description
    '''

    def __init__(self, stemming=True, lem=False):

        english_stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords = [self.remove_accent(sw) for sw in english_stopwords]

        # Args
        self.stemming = stemming
        self.lem = lem


    @staticmethod
    def convert_text_to_lower_case(txt):
        '''
        :param txt: text (str) to put in lower case
        :return: text (str) in lower
        '''
        return txt.lower()

    @staticmethod
    def remove_accent(txt):
        '''
        :param txt: text (str) to remove accent
        :return: text (str) without accent
        '''
        return unicodedata.normalize('NFD', txt).encode('ascii', 'ignore').decode("utf-8")

    @staticmethod
    def remove_non_letters(txt):
        '''
        :param txt: text (str)
        :return: only text (str) with alphabet letters
        '''
        return re.sub('[^a-z_]', ' ', txt)

    def remove_stopwords(self, txt):
        '''
        :param txt: text (str)
        :return: (list of str) without stopwords
        '''
        return [w for w in txt.split() if (w not in self.stopwords)]

    def get_stem(self, tokens):
        '''
        :param tokens: (list of str)
        :return: (list of str) with stemmed tokens
        '''
        stemmer = nltk.stem.SnowballStemmer('english')
        return [stemmer.stem(token) for token in tokens]

    def get_lem(self, tokens):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    def apply_all_transformation(self, txt):
        '''
        apply all transformation above on a text description
        :param txt: text description (str)
        :return: (list of str) cleaned
        '''
        txt = self.convert_text_to_lower_case(txt)
        txt = self.remove_accent(txt)
        txt = self.remove_non_letters(txt)
        tokens = self.remove_stopwords(txt)
        if self.stemming:
            tokens = self.get_stem(tokens)
        if self.lem:
            tokens = self.get_lem(tokens)
        return tokens


    def clean_df_column(self, df, column_name, clean_column_name):
        '''
        clean all the text lines of the df columns and add a new columns on this df that contains the cleaned line
        :param df:
        :param column_name: column name to clean (str)
        :param clean_column_name: cleaned column name (str)
        :return: df with cleaned column
        '''
        df[clean_column_name] = [" ".join(self.apply_all_transformation(x)) for x in tqdm(df[column_name].values)]
