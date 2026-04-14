# %%
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import re
import string
import warnings
warnings.filterwarnings("ignore")

# %%
nltk.download("punkt")

# %%
dataset = pd.read_csv("twitter_training.csv", header=None)

dataset.sample(10)

# %%
dataset.rename(columns = {1: "Product", 2: "Sentiment", 3: "Text"}, inplace = True)


dataset.sample(10)

# %%
dataset.drop(columns= 0, inplace = True)


# %%
dataset.head()

# %%
dataset["Sentiment"].value_counts()

# %%
encoder = LabelEncoder()

dataset["Sentiment"] = encoder.fit_transform(dataset["Sentiment"])

dataset.head()


# %%
labels = encoder.inverse_transform(dataset["Sentiment"].value_counts().index)
encoder.classes_

# %%
plt.bar(labels, dataset["Sentiment"].value_counts().values, color = ["red", "green", "blue" , "yellow"])
plt.ylabel("Count")
plt.title("Distribution of Sentiment")
plt.show()

# %%
plt.pie(dataset["Sentiment"].value_counts().values, labels = labels, colors = ["red", "green", "blue" , "yellow"], autopct = "%1.1f%%")
plt.title("Distribution of Sentiment")
plt.show()

# %% [markdown]
# ## EDA

# %%
dataset["num_characters"] = dataset["Text"].astype(str).apply(len)

# %%
dataset 

# %%
# num of words
dataset["num_words"] = dataset["Text"].astype(str).apply(lambda x: len(nltk.word_tokenize(x)))

# %%
dataset

# %%
# num of sentences
dataset["num_sentences"] = dataset["Text"].astype(str).apply(lambda x: len(nltk.sent_tokenize(x)))

# %%
dataset 

# %%
dataset[["num_characters", "num_words", "num_sentences"]].describe()    

# %%
# details of the text with Irrelevant sentiment
dataset[dataset["Sentiment"] == 0][["num_characters", "num_words", "num_sentences"]].describe()

# %%
#details of the text with negative sentiment
dataset[dataset["Sentiment"] == 1][["num_characters", "num_words", "num_sentences"]].describe()

# %%
#details of the text with neutral sentiment
dataset[dataset["Sentiment"] == 2][["num_characters", "num_words", "num_sentences"]].describe()


# %%
# details of the text with positive sentiment
dataset[dataset["Sentiment"] == 3][["num_characters", "num_words", "num_sentences"]].describe()


# %%
sns.histplot(dataset[dataset["Sentiment"] == 0]["num_characters"])
sns.histplot(dataset[dataset["Sentiment"] == 1]["num_characters"], color= "red")
sns.histplot(dataset[dataset["Sentiment"] == 2]["num_characters"], color = "yellow")
sns.histplot(dataset[dataset["Sentiment"] == 3]["num_characters"], color = "green")



# %%
sns.histplot(dataset[dataset["Sentiment"] == 0]["num_words"])
sns.histplot(dataset[dataset["Sentiment"] == 1]["num_words"], color= "red")
sns.histplot(dataset[dataset["Sentiment"] == 2]["num_words"], color = "yellow")
sns.histplot(dataset[dataset["Sentiment"] == 3]["num_words"], color = "green")

# %%
sns.pairplot(dataset, hue = "Sentiment")

# %%
sns.heatmap(dataset.select_dtypes(include=['number']).corr(), annot= True)

# %% [markdown]
# # Data Preprocessing
# 
# lowercase
# 
# tokenisation
# 
# Removing special characters
# 
# Removing stop words and punctuation
# 
# Stemming

# %%
# defining a function to convert text to lowercase, tokenise the text, and remove stop words, punctuation and special characters from the text and stem the text
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re   

ps = PorterStemmer()

def transform_text(text):
    text = str(text).lower()
    
    # REMOVE URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    text = nltk.word_tokenize(text)
    
    y = []
    
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words("english") and i not in ["@", "#"]:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# %%
dataset["transformed_text"] = dataset["Text"].apply(transform_text)
dataset.head()

# %%
# generating word cloud for all kind of sentiments
from wordcloud import WordCloud
wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color = "white")

Irrelevant_wc = wc.generate(dataset[dataset["Sentiment"] == 0]["transformed_text"].str.cat(sep = " "))
negative_wc = wc.generate(dataset[dataset["Sentiment"] == 1]["transformed_text"].str.cat(sep = " "))
neutral_wc = wc.generate(dataset[dataset["Sentiment"] == 2]["transformed_text"].str.cat(sep = " "))
positive_wc = wc.generate(dataset[dataset["Sentiment"] == 3]["transformed_text"].str.cat(sep = " "))
plt.figure(figsize = (10, 10))
plt.imshow(Irrelevant_wc)
plt.axis("off")
plt.title("Irrelevant Sentiment Word Cloud")
plt.show()
plt.figure(figsize = (10, 10))
plt.imshow(negative_wc)
plt.axis("off")
plt.title("Negative Sentiment Word Cloud")
plt.show()
plt.figure(figsize = (10, 10))
plt.imshow(neutral_wc)
plt.axis("off")
plt.title("Neutral Sentiment Word Cloud")
plt.show()
plt.figure(figsize = (10, 10))
plt.imshow(positive_wc)
plt.axis("off")
plt.title("Positive Sentiment Word Cloud")
plt.show()


# %%
irrelevant_corpus = []
irrelevant_sent = dataset[dataset["Sentiment"] == 0]["transformed_text"]
for i in irrelevant_sent:
    for j in i.split():
        irrelevant_corpus.append(j)
len(irrelevant_corpus)

# %%
# negative corpus
negative_corpus = []
negative_sent = dataset[dataset["Sentiment"] == 1]["transformed_text"]
for i in negative_sent:
    for j in i.split():
        negative_corpus.append(j)
len(negative_corpus)

# %%
# neutral corpus
neutral_corpus = []
neutral_sent = dataset[dataset["Sentiment"] == 2]["transformed_text"]
for i in neutral_sent:      
    for j in i.split():
        neutral_corpus.append(j)    
len(neutral_corpus)

# %%
#positive corpus
positive_corpus = []
positive_sent = dataset[dataset["Sentiment"] == 3]["transformed_text"]
for i in positive_sent:
    for j in i.split():
        positive_corpus.append(j)
len(positive_corpus)

# %%
from collections import Counter
Counter(irrelevant_corpus).most_common(500)


# %%
Counter(negative_corpus).most_common(500)

# %%
Counter(neutral_corpus).most_common(500)

# %%
Counter(positive_corpus).most_common(500)

# %%
data_train, data_test = train_test_split(dataset, test_size = 0.2, random_state = 42, stratify= dataset["Sentiment"])

# %%
data_train.shape

# %%
data_test.shape

# %%

# Define parameters clearly
VOCAB_SIZE = 5000
MAX_LEN = 200
OOV_TOKEN = "<OOV>"

# Initialize tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)

# Fit ONLY on training data (correct 👍)
tokenizer.fit_on_texts(data_train["transformed_text"])

# Convert text → sequences
train_sequences = tokenizer.texts_to_sequences(data_train["transformed_text"])
test_sequences = tokenizer.texts_to_sequences(data_test["transformed_text"])

# Pad sequences (explicit settings)
X_train = pad_sequences(
    train_sequences,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)

X_test = pad_sequences(
    test_sequences,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)



# %%
y_train = data_train["Sentiment"].values
y_test = data_test["Sentiment"].values


# %%
# Build the model
model = Sequential()
model.add(Embedding(VOCAB_SIZE, output_dim= 128, input_length=MAX_LEN))
model.add(LSTM(128, dropout= 0.2, recurrent_dropout= 0.2))
model.add(Dense(4, activation="softmax"))

# %%
model.summary()

# %%
# complile the mode
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# %%
early_stop = EarlyStopping(
    monitor='val_loss',     # watch validation loss
    patience=2,             # stop if no improvement for 2 epochs
    restore_best_weights=True)

model.fit(X_train, y_train, epochs = 10, validation_split= 0.2, callbacks=[early_stop])

# %%
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


