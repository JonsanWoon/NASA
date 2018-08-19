import MySQLdb
from keras.datasets import imdb
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from stemming.porter2 import stem
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import tensorflow as tf
import numpy as np
import connection


def connect_database():
    db = MySQLdb.connect(host='127.0.0.1', db='news10', user='root', passwd='mysql')
    return db


def import_data():
    db = connect_database()
    c = db.cursor(MySQLdb.cursors.DictCursor)
    c.callproc("content")
    content = c.fetchall()
    data = []
    for rows in content:
        data.append(rows)
    c.close()
    return data


def import_sentiment():
    db = connect_database()
    c = db.cursor(MySQLdb.cursors.DictCursor)
    c.callproc("results")
    result = c.fetchall()
    sentiment = []
    for rows in result:
        sentiment.append(rows)
    c.close()
    return sentiment


def remove_html(tags):
    remove = re.compile('<.*?>')
    clean_sentence = re.sub(remove, '', str(tags))
    return clean_sentence


def tokenize(words):
    tokenize_words = word_tokenize(words)
    return tokenize_words


def remove_common_words(filter_words):
    db = connect_database()
    c = db.cursor(MySQLdb.cursors.DictCursor)
    c.callproc("getSetting", ["RmvCommonWord"])
    result = c.fetchone()
    c.close()
    if result["Value"] == "1":
        stop_words = set(stopwords.words('english'))
        filtered_sentence = [w for w in filter_words if not w in stop_words]

        for w in filter_words:
            if w not in stop_words:
                filtered_sentence.append(w)
            return filtered_sentence
    else:
        return filter_words


def stemmed_word(words):
    db = connect_database()
    c = db.cursor(MySQLdb.cursors.DictCursor)
    c.callproc("getSetting", ["wordStemming"])
    result = c.fetchone()
    c.close()
    stemmed_words = []
    if result["Value"] == "1":
        for w in words:
            stemmed_words.append(stem(w))
    else:
        for w in words:
            stemmed_words.append(w)
    return stemmed_words


def training(data, result):
    db = connect_database()
    c = db.cursor(MySQLdb.cursors.DictCursor)
    c.callproc("getSetting", ["randomstate"])
    randomstate = c.fetchone()
    randomstate = int(randomstate['Value'])
    c.close()

    db = connect_database()
    c = db.cursor(MySQLdb.cursors.DictCursor)
    c.callproc("getSetting", ["nb_epoch"])
    nb_epoch = c.fetchone()
    nb_epoch = int(nb_epoch['Value'])
    c.close()

    db = connect_database()
    c = db.cursor(MySQLdb.cursors.DictCursor)
    c.callproc("getSetting", ["train_verbose"])
    train_verbose = c.fetchone()
    train_verbose = int(train_verbose['Value'])
    c.close()

    db = connect_database()
    c = db.cursor(MySQLdb.cursors.DictCursor)
    c.callproc("getSetting", ["evaluate_verbose"])
    evaluate_verbose = c.fetchone()
    evaluate_verbose = int(evaluate_verbose['Value'])
    c.close()

    df = pd.DataFrame({"news": [x["Content"] for x in data], "sentiments": [x["Result"] for x in result]})
    df['news'] = [x.lower() for x in df['news']]
    df['news'] = df['news'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
    df['news'] = df['news'].apply((lambda x: remove_html(x)))
    df['news'] = df['news'].apply((lambda x: tokenize(x)))
    df['news'] = df['news'].apply((lambda x: remove_common_words(x)))
    tokenizer = Tokenizer(nb_words=2500, split=' ')
    tokenizer.fit_on_texts(df['news'].values)

    X = tokenizer.texts_to_sequences(df['news'].values)
    X = pad_sequences(X)
    Y = pd.get_dummies(df['sentiments']).values

    embed_dim = 128
    lstm_out = 300
    batch_size = 32

    model = Sequential()
    model.add(Embedding(2500, embed_dim, input_length=X.shape[1], dropout=0.1))
    model.add(LSTM(lstm_out, dropout_U=0.1, dropout_W=0.1))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=randomstate)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=train_verbose)

    score, acc = model.evaluate(X_valid, Y_valid, verbose=evaluate_verbose, batch_size=batch_size)
    print("Logloss score: %.2f" %(score))
    print("Validation set Accuracy: %.2f" % (acc))
    model.save('sentiment_analysis_engine')
    return model


def prediction(X):
    print(X)
    tokenizer = Tokenizer(nb_words=2500, split=' ')
    tokenizer.fit_on_texts(X)

    news_article = tokenizer.texts_to_sequences(X)
    news_article = pad_sequences(news_article, maxlen=671)

    model = load_model('sentiment_analysis_engine')
    predict = model.predict(news_article, batch_size=1, verbose=2)
    print(predict)
    print(np.argmax(predict))

    if np.argmax(predict) <= 0.5:
        return "-1"

    elif np.argmax(predict) >= 0.5:
        return "1"


def port_listener():
    db = connect_database()
    c = db.cursor(MySQLdb.cursors.DictCursor)
    c.callproc("getSetting", ["port"])
    port = c.fetchone()
    port = int(port['Value'])
    c.close()

    connection.define('port', default=port, help='run on the given port', type=int)  # define the setting such as port
    application = connection.tornado.web.Application([(r'/', connection.RequestHandler, dict(password="ASD")), ])
    http_server = connection.tornado.httpserver.HTTPServer(application)  # run the application
    http_server.listen(connection.options.port)  # Identify which port to listen
    connection.tornado.ioloop.IOLoop.current().start()  # start the service


def main():
    data = import_data()
    result = import_sentiment()
    training(data, result)
    port_listener()
    prediction("<html>Ths is very bad for testing</html>")


if __name__ == "__main__":
    main()
