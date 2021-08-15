from collections import Counter
import re
import string

from keras.preprocessing.text import Tokenizer
import nltk
import nltk.corpus as corpus
import nltk.stem as stem
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

# nltk.download('stopwords')
# nltk.download('wordnet')


def main():
    # load data
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')
    print("Shape of train: ", train.shape)
    print("Shape of test: ", test.shape)
    print("Head of train: \n", train.head(3))

    # data processing
    train['text'] = train['text'].apply(lambda s: preprocess(s))
    test['text'] = test['text'].apply(lambda s: preprocess(s))
    vocab = create_vocab(train)
    train['text'] = train['text'].apply(lambda s: filter_words(s, vocab))
    test['text'] = test['text'].apply(lambda s: filter_words(s, vocab))
    print("Head of train editted: \n", train.head(3))

    # tokenize
    tokenizer = Tokenizer(num_words=len(vocab))
    tokenizer.fit_on_texts(train['text'])
    word_index = tokenizer.word_index
    train_text = tokenizer.texts_to_matrix(train['text'], mode='freq')
    print(train_text.shape)

    # split data
    X_train, X_val, y_train, y_val = train_test_split(train_text, train['target'], test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train.values))
    val_ds = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val.values))
    train_loader = DataLoader(train_ds, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=100, shuffle=True)
    pass


def preprocess(tweet):
    stopwords = list(corpus.stopwords.words('english'))
    punctuations = list(string.punctuation)
    lemmatizer = stem.WordNetLemmatizer()

    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)  # removing urls
    tweet = re.sub('[^\w]', ' ', tweet)  # remove embedded special characters in words (for example #earthquake)
    tweet = re.sub('[\d]', '', tweet)  # this will remove numeric characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    words = tweet.lower().split()

    sentence = ''
    for word in words:
        if word in stopwords + punctuations:
            continue
        word = lemmatizer.lemmatize(word, pos='v')

        if len(word) <= 3:
            continue
        sentence += word + ' '
    return sentence


def create_vocab(df):
    vocab = Counter()
    for i in range(df.shape[0]):
        vocab.update(df.loc[i, 'text'].split())
    return {x: count for x, count in vocab.items() if count >= 2}


def filter_words(text, vocab):
    return ' '.join([word for word in text.split() if word in vocab])


if __name__ == '__main__':
    main()
