import nltk
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras import backend as K

tagged_sentences = nltk.corpus.treebank.tagged_sents()

sentences, sentence_tags = [], []
for tagged_sentence in tagged_sentences:
	sentence, tags = zip(*tagged_sentence)
	sentences.append(np.array(sentence))
	sentence_tags.append(np.array(tags))


(train_sentences,
 test_sentences,
 train_tags,
 test_tags) = train_test_split(sentences, sentence_tags, test_size = 0.2)


words, tags = set([]), set([])
for s in train_sentences:
	for w in s:
		words.add(w.lower())
for ts in train_tags:
	for t in ts:
		tags.add(t)
word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0
word2index['-OOV-'] = 1
tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0


train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
for s in train_sentences:
	s_int = []
	for w in s:
		try:
			s_int.append(word2index[w.lower()])
		except KeyError:
			s_int.append(word2index['-OOV-'])
	train_sentences_X.append(s_int)
for s in test_sentences:
	s_int = []
	for w in s:
		try:
			s_int.append(word2index[w.lower()])
		except KeyError:
			s_int.append(word2index['-OOV-'])
	test_sentences_X.append(s_int)
for s in train_tags:
	train_tags_y.append([tag2index[t] for t in s])
for s in test_tags:
	test_tags_y.append([tag2index[t] for t in s])


MAX_LENGTH = len(max(train_sentences_X, key=len))


train_sentences_X = pad_sequences(train_sentences_X, maxlen = MAX_LENGTH, padding = 'post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen = MAX_LENGTH, padding = 'post')
train_tags_y = pad_sequences(train_tags_y, maxlen = MAX_LENGTH, padding = 'post')
test_tags_y = pad_sequences(test_tags_y, maxlen = MAX_LENGTH, padding = 'post')



def to_categorical(sequences, categories):
	cat_sequences = []
	for s in sequences:
		cats = []
		for item in s:
			cats.append(np.zeros(categories))
			cats[-1][item] = 1.0
		cat_sequences.append(cats)
	return np.array(cat_sequences)
 

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
print(cat_train_tags_y[0])


def logits_to_tokens(sequences, index):
	token_sequences = []
	for categorical_sequence in sequences:
		token_sequence = []
		for categorical in categorical_sequence:
			token_sequence.append(index[np.argmax(categorical)])
		token_sequences.append(token_sequence)
	return token_sequences


def ignore_class_accuracy(to_ignore=0):
	def ignore_accuracy(y_true, y_pred):
		y_true_class = K.argmax(y_true, axis=-1)
		y_pred_class = K.argmax(y_pred, axis=-1)
		ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
		matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32')
		accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
		return accuracy
	return ignore_accuracy


model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy', ignore_class_accuracy(0)])
model.summary()

model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=10, validation_split=0.2)


scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")


test_samples = [
	'running is very important for me .'.split(),
	'I was running every day for a month .'.split()
]
test_samples_X = []
for s in test_samples:
	s_int = []
	for w in s:
		try:
			s_int.append(word2index[w.lower()])
		except KeyError:
			s_int.append(word2index['-OOV-'])
	test_samples_X.append(s_int)
test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')
print(test_samples_X)


print('hard to read predictions:')
predictions = model.predict(test_samples_X)
print(predictions, predictions.shape)

print('reverse predictions:')
print(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))
