print('import')
import pickle
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
 
classifier = Pipeline([
	('vectorizer', DictVectorizer(sparse=False)),
	('classifier', DecisionTreeClassifier(criterion='entropy'))
	])

print('loading')
with open ('1000_sents', 'rb') as f:
	sents = pickle.load(f)

def features(sentence, index):
	return {
		'word': sentence[index],
		'is_first': index == 0,
		'is_last': index == len(sentence) - 1,
		'is_capitalized': sentence[index][0].upper() == sentence[index][0],
		'is_all_caps': sentence[index].upper() == sentence[index],
		'is_all_lower': sentence[index].lower() == sentence[index],
		'prefix-1': sentence[index][0],
		'prefix-2': sentence[index][:2],
		'prefix-3': sentence[index][:3],
		'suffix-1': sentence[index][-1],
		'suffix-2': sentence[index][-2:],
		'suffix-3': sentence[index][-3:],
		'prev_word': '' if index == 0 else sentence[index - 1],
		'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
		'has_hyphen': '-' in sentence[index],
		'is_numeric': sentence[index].isdigit(),
		'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
	}

def untag(tagged_sentence):
	return [w[0] for w in tagged_sentence]

print('preparing')
random.shuffle(sents)

cutoff = int(0.9 * len(sents))
train_sents = sents[:cutoff]
test_sents = sents[cutoff:]

def transform_to_dataset(tagged_sentences):
	X, y = [], []
	for tagged in tagged_sentences:
		for index in range(len(tagged)):
#			print('index   ', index)
#			print('tagged   ', tagged)
#			print('tagged[index]   ', tagged[index])
#			print('tagged[index][1]   ', tagged[index][1])
			X.append(features(untag(tagged), index))
			y.append(tagged[index][1])
	return X, y
 
Features, tags = transform_to_dataset(train_sents)


print('training')
classifier.fit(Features, tags)

print('testing')
Features_test, tags_test = transform_to_dataset(test_sents)
print("Accuracy: ", classifier.score(Features_test, tags_test))

pred = classifier.predict(Features_test)

print('Precision None: ', metrics.precision_score(tags_test, pred, average = None))
print('Precicion macro: ', metrics.precision_score(tags_test, pred, average = 'macro'))
print(confusion_matrix(tags_test, pred))



#print('tagging')
#def pos_tag(sentence):
#	tag = classifier.predict([features(sentence, index) for index in range(len(sentence))])
#	return zip(sentence, tag)

#testing = []
#for sent in test_sents:
#	testing.append(untag(sent))
#print(testing)
#for sent in testing:
#	print(list(pos_tag(sent)))
