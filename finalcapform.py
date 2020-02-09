from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import collections

from random import randrange

import pandas as pd
import numpy as np
import pickle


num_art = 400 # or 500
cluster_map_new = pd.read_pickle(f'cluster_map_new_{num_art}.pkl')

for num in [1]: #range(cluster_map_new['KNN_Clusters'].nunique()): 
    rnn_size = 768 # size of RNN
    seq_length = 4 # sequence length
    learning_rate = 0.001 #learning rate
    batch_size = 250 # batch size
    num_epochs = 20 # number of epochs

    print("Cluster Number: ", num)
    print(' ')
    new_df = cluster_map_new.loc[cluster_map_new['KNN_Clusters']==num]
    print("New Dataframe Shape: ", new_df.shape)
    print(' ')

    wordlist = []

    for cont in new_df['Content_cl'].values.tolist():
        wordlist += cont
    
    print("Size of Wordlist: ", len(wordlist))
    print(' ')

    word_counts = collections.Counter(wordlist)

    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    vocab = {x: i for i, x in enumerate(vocabulary_inv)}
    words = [x[0] for x in word_counts.most_common()]

    vocab_size = len(words) 
    print("Vocab Size: ", vocab_size)
    print(' ')

    vocab_dict = {}
    vocab_dict[f"cluster_num_{num}_{num_art}"] = vocab_size

    sequences = []
    next_words = []
    for i in range(0, len(wordlist) - seq_length, 1):
        sequences.append(wordlist[i: i + seq_length])
        next_words.append(wordlist[i + seq_length])
    
    X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
    y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, word in enumerate(sentence):
            X[i, t, vocab[word]] = 1
        y[i, vocab[next_words[i]]] = 1

    output = open(f'vocab_dict_clust_{num}_{num_art}.pkl', 'wb')
    pickle.dump(vocab_dict, output)
    output.close()
    print('Successful Save')
    
    output = open(f'vocabulary_clust_{num}_{num_art}.pkl', 'wb')
    pickle.dump(vocab, output)
    output.close()
    print('Successful Save')

    output = open(f'vocabulary_inv_clust_{num}_{num_art}.pkl', 'wb')
    pickle.dump(vocabulary_inv, output)
    output.close()
    print('Successful Save')

    print('Build LSTM model.')
    md = Sequential()
    md.add(LSTM(rnn_size, activation="relu", return_sequences=True,input_shape=(seq_length, vocab_size)))
    md.add(Dropout(0.4))
    md.add(LSTM(rnn_size, return_sequences=True))
    md.add(Dropout(0.4)) 
    md.add(LSTM(rnn_size))
    md.add(Dropout(0.4))
    md.add(Dense(vocab_size, activation='softmax'))
    md.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
    print("model built!")

    print(md.summary())
    print(' ')
    # define the checkpoint
    filepath = f"saved-model-clust-{num}_{num_art}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]
    history = md.fit(X, y, batch_size=batch_size, shuffle=True, epochs=num_epochs, validation_split=0.1, callbacks=callbacks_list)

    md.save(f"model_clust_{num}_{num_art}.h5")
    print("Saved model to disk")