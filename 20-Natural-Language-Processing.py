import nltk
nltk.download_shell()

messages = [line.rstrip() for line in open('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\20-Natural-Language-Processing\\smsspamcollection\\SMSSpamCollection')]
print(len(messages))

for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')

import pandas as pd
messages = pd.read_csv('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\20-Natural-Language-Processing\\smsspamcollection\\SMSSpamCollection', sep = '\t', names = ['label', 'message'])
messages.head()

# EDA
messages.describe()
messages.groupby('label').describe()

messages['length'] = messages['message'].apply(len)
messages.head()

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

messages['length'].plot(bins = 50, kind = 'hist')
sns.displot(messages['length'])
plt.show()

messages.length.describe()
messages[messages['length'] == 910]['message'].iloc[0]

messages.hist(column = 'length', by = 'label', bins = 50, figsize =(12,4))
plt.show() # longer messages tend to be spam

# Text Pre-processing - Normalisation
    # Need to transfor text values into some sort of numerical feature to perform a classification task
    # 'Bag of words' each unique word in the text will be represented by a number
    
import string
# Get rid of punctuation
mess = 'Sample Message! Notice: it has punctuation.'
nopunc = [c for c in mess if c not in string.punctuation]

nopunc = ''.join(nopunc)
nopunc
# get rid of stopwords (very common words)
from nltk.corpus import stopwords
stopwords.words('english')[0:10]

nopunc.split()

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    nopunc = [c for c in mess if c not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

messages.head()

messages['message'].head(5).apply(text_process)

# Vectorisation

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))

message_4 = messages['message'][3]
print(message_4)

# 7 unique words, 2 repeat twice
bow4 = bow_transformer.transform([message_4])
print(bow4)
print(bow4.shape)

print(bow_transformer.get_feature_names_out()[4068])
print(bow_transformer.get_feature_names_out()[9554])

# Repeat on all messages
messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(sparsity))

# TF-IDF

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(messages_bow)

tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

# Train a Model - Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])

#evaluating accuracy on the same data used fro training - Never do
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)

# Train/Test Split

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)

# Data Pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report
print(classification_report(predictions,label_test))




