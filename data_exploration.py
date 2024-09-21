import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the cleaned data
df = pd.read_csv('cleaned_data_utf8.csv')
print(df.info())

# Drop rows with NaN values in 'text_cleaned'
df.dropna(subset=['text_cleaned'], inplace=True)

# Verify that there are no NaN values left
print("Number of NaN values in 'text_cleaned':", df['text_cleaned'].isnull().sum())

# Text representation using TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['text_cleaned'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
print(tfidf_df.shape)
print(tfidf_df.head())

# Visualize the class distribution of the target variable
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='target', palette='viridis')
plt.title('Class Distribution of Target Variable')
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Count occurrences of each class in the target variable
class_distribution = df['target'].value_counts().reset_index()
class_distribution.columns = ['Target Class', 'Count']

print(class_distribution)

# Create a table to show class distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=class_distribution.values, colLabels=class_distribution.columns, cellLoc='center', loc='center')

plt.title('Class Distribution of Target Variable')
plt.show()

# Group by target variable to get vocabulary sizes
vocabulary_sizes = df.groupby('target')['text_cleaned'].apply(lambda x: ' '.join(x).split()).apply(set).apply(len).reset_index()
vocabulary_sizes.columns = ['Target Class', 'Vocabulary Size']

# Display the vocabulary sizes
print(vocabulary_sizes)

plt.figure(figsize=(10, 6))
plt.bar(vocabulary_sizes['Target Class'].astype(str), vocabulary_sizes['Vocabulary Size'], color='skyblue')
plt.title('Vocabulary Sizes in Each Class')
plt.xlabel('Target Class')
plt.ylabel('Vocabulary Size')
plt.xticks(rotation=0)
plt.show()

# Add the target column to the TF-IDF DataFrame
tfidf_df['target'] = df['target'].values

def get_top_tfidf_words(df, target_col, top_n=10):
    results = {}
    for target_class in df[target_col].unique():
        class_df = df[df[target_col] == target_class]
        mean_tfidf = class_df.drop(columns=[target_col]).mean().sort_values(ascending=False)
        results[target_class] = mean_tfidf.head(top_n)
    return results

top_tfidf_words = get_top_tfidf_words(tfidf_df, 'target', top_n=10)

for target_class, words in top_tfidf_words.items():
    words.plot(kind='bar', color='skyblue')
    plt.title(f'Top TF-IDF Words for Class {target_class}')
    plt.xlabel('Words')
    plt.ylabel('Mean TF-IDF Score')
    plt.xticks(rotation=45)
    plt.show()
