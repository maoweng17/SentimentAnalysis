## ====================================== Import Package ====================================== ##
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

# use matplotlib, seaborn package to draw charts
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import seaborn as sns

# NLP preprocessing
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

# machine learning modelling
from sklearn.model_selection import train_test_split
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, SpatialDropout1D,Convolution1D, GlobalMaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers.convolutional import Conv1D
from keras.callbacks import EarlyStopping


## ===================================== Data Preprocessing ===================================== ##
df_master = pd.read_csv("imdb_master.csv", encoding='latin-1', index_col = 0)
df_master[['id','rating']] = df_master.file.str.replace('.txt','').str.split('_',expand=True)
df_master['id'] = pd.to_numeric(df_master['id'], errors='coerce')
df_master['rating'] = pd.to_numeric(df_master['rating'])
df_master = df_master[df_master.label != 'unsup']
df_master['review_length'] = df_master.review.apply(lambda x: len(x.split(" ")))
df_master.head()


# --------------------------------- Exploratory Data Analysis ----------------------------------- #
# For chart prupose: Split the dataframe by target
pos = df_master[df_master.label == 'pos']
neg = df_master[df_master.label == 'neg']

# Set Style
sns.set_style("whitegrid")


# imbalanced data ? No
pd.value_counts(df_master.label).plot.bar()


# Visualise: Word Counts of Review - Distribution
df_master.groupby(['label'])[['rating']].agg('mean').reset_index()

f, ax_hist = plt.subplots(1,figsize=(16, 6))
plt.title("Word Counts of Review - Distribution")

sns.distplot(pos[['review_length']], hist=False, rug=True)
sns.distplot(neg[['review_length']], hist=False, rug=True)


# ------------------------------------ clean text -------------------------------------- #
# Not remove stopwords: to avoid the meaning of sentence changes (e.g. I didn't like it -> I like it -> positive?)

def cleanText(review):
    # 1. remove HTML line break tag: <br />
    review_text = review.replace("<br />"," ")
    # 2. convert words to lower case
    review_text = review_text.lower()
    # 3. lemmatization with NLTK
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(i) for i in review_text.split(" ")]
    review_text = " ".join(words)
    
    return review_text

df_master['cleanedReview'] = df_master['review'].apply(lambda x: cleanText(x))
# df_master.head()

# 4. keep 5000 top frequent words in training, convert text to sequence
max_features = 5000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df_master[df_master.type == 'train']['cleanedReview'])
df_master['cleanedReviewSequence'] = tokenizer.texts_to_sequences(df_master['cleanedReview'])

#df_master.head()

## =============================== Modelling: Sentiment Analysis =============================== ##

# -------------------------------------- Split Dataset ---------------------------------------- #
maxlen = 300

df_master['label_num'] = df_master['label'].map(dict(neg=0, pos=1))
train_X = df_master[df_master.type == 'train']['cleanedReviewSequence']
X_test  = df_master[df_master.type == 'test']['cleanedReviewSequence']
train_y = df_master[df_master.type == 'train']['label_num'].values
y_test  = df_master[df_master.type == 'test']['label_num'].values

train_X = sequence.pad_sequences(train_X, maxlen=maxlen,padding='post',truncating='post')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen,padding='post',truncating='post')

# Extract some part of data as validation datasets to tune model
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.3, random_state=42)



# --------------------------------------- LSTM Model ----------------------------------------- #
# ------------- Parameters ------------- #
embedding_dims = 50
batch_size = 32
nb_epoch = 4

# ------------ Model define ------------ #
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))

model.add(LSTM(100))

          
model.add(Dense(2, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

# ------------ Fit model ------------ #
model.fit(X_train, 
          y_train, 
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(X_valid, y_valid))


loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("Training: accuracy = %f  ;  loss = %f" % (accuracy, loss))
loss, accuracy = model.evaluate(X_valid, y_valid, verbose=0)
print("Validation: accuracy = %f  ;  loss = %f" % (accuracy, loss))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))


# --------------------------------------- CNN Model ----------------------------------------- #

# ------------- Parameters ------------- #
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 4

# ------------ Model define ------------ #
model_cnn = Sequential()
embedding_layer = Embedding(max_features,
                           embedding_dims,
                           input_length=maxlen)
model_cnn.add(embedding_layer)
model_cnn.add(SpatialDropout1D(0.5))
model_cnn.add(Conv1D(filters=nb_filter, activation="relu", kernel_size=3, strides=1, padding="valid"))


model_cnn.add(GlobalMaxPooling1D())

model_cnn.add(Dense(hidden_dims))
model_cnn.add(Dropout(0.5))
model_cnn.add(Activation('relu'))

model_cnn.add(Dense(1))
model_cnn.add(Activation('sigmoid'))
print(model_cnn.summary())

model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #learning rate
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# ------------ Fit model ------------ #
h = model_cnn.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_data=(X_valid, y_valid),
    verbose=1,
    callbacks=[es])

loss, accuracy = model_cnn.evaluate(X_train, y_train, verbose=0)
print("Training: accuracy = %f  ;  loss = %f" % (accuracy, loss))
loss, accuracy = model_cnn.evaluate(X_valid, y_valid, verbose=0)
print("Validation: accuracy = %f  ;  loss = %f" % (accuracy, loss))
loss, accuracy = model_cnn.evaluate(X_test, y_test, verbose=0)
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

