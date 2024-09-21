import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt') 
nltk.download('stopwords')
nltk.download('wordnet')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('data_to_be_cleansed.csv')
# Exploratory Data Analysis
print(df.shape)
print(df.info())
print(df.dtypes)

duplicate_rows_df = df[df.duplicated()]
print("Number of duplicate rows: ", duplicate_rows_df.shape)

print(df.count())
print(df.isnull().sum())
df = df.dropna()
print(df.count())
print(df.isnull().sum())

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        cleaned_text = ' '.join(tokens)
        return cleaned_text
    else:
        return ''

df['text_cleaned'] = df['text'].apply(clean_text)
print(df.info())
df = df.drop(df.columns[[0, 1, 2]], axis=1)
df = df.dropna(subset=['text_cleaned'])
df = df[['text_cleaned', 'target']]
# Ensure there are no NaN values in 'text_cleaned'
df['text_cleaned'].dropna(inplace=True)
print("Number of NaN values in 'text_cleaned':", df['text_cleaned'].isnull().sum())
print(df.info())
df.to_csv('cleaned_data_utf8.csv', encoding='utf-8')
