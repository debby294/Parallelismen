print('import')
import pickle
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

classifier1 = Pipeline([
	('vectorizer', DictVectorizer(sparse=False)),
	('classifier', RandomForestClassifier()) #default: n_estimators=10, max_depth=None, random_state=None --> Accuracy %	88,5	88,5	88,5	88,7	88,4
	])
classifier2 = Pipeline([
	('vectorizer', DictVectorizer(sparse=False)),
	('classifier', RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)) #Accuracy %	53,5	55,8	52,7	55,4	55,8
	])
classifier3 = Pipeline([
	('vectorizer', DictVectorizer(sparse=False)),
	('classifier', RandomForestClassifier(max_depth=100)) #Accuracy %	88,5	89,2	89	87,1	88,3
	])
classifier4 = Pipeline([
	('vectorizer', DictVectorizer(sparse=False)),
	('classifier', RandomForestClassifier(max_depth=200)) #Accuracy %	89,2	88,3	88,9	89	88,6
	])
classifier5 = Pipeline([
	('vectorizer', DictVectorizer(sparse=False)),
	('classifier', RandomForestClassifier(max_depth=500)) #Accuracy %	89	88,7	88	87,8	88
	])
classifier6 = Pipeline([
	('vectorizer', DictVectorizer(sparse=False)),
	('classifier', RandomForestClassifier(max_depth=1000)) #Accuracy %	89,2	88,9	88,4	89,4	88,1
	])


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

def transform_to_dataset(tagged_sentences):
	X, y = [], []
	for tagged in tagged_sentences:
		for index in range(len(tagged)):
			X.append(features(untag(tagged), index))
			y.append(tagged[index][1])
	return X, y


print('loading')
with open ('dta_komplett_tcf-full-all-10000sents', 'rb') as f:
	sents = pickle.load(f)


print('preparing')
random.shuffle(sents)

cutoff = int(0.9 * len(sents))
train_sents = sents[:cutoff]
test_sents = sents[cutoff:]
 
Features, tags = transform_to_dataset(train_sents)

#print('training 1')
#classifier1.fit(Features, tags)
#print('training 2')
#classifier2.fit(Features, tags)
#print('training 3')
#classifier3.fit(Features, tags)
#print('training 4')
#classifier4.fit(Features, tags)
#print('training 5')
#classifier5.fit(Features, tags)
print('training 6')
classifier6.fit(Features, tags)


print('testing')
Features_test, tags_test = transform_to_dataset(test_sents)
#print("Accuracy 1 (default):", classifier1.score(Features_test, tags_test))
#print("Accuracy 2 (10):", classifier2.score(Features_test, tags_test))
#print("Accuracy 3 (100):", classifier3.score(Features_test, tags_test))
#print("Accuracy 4 (200):", classifier4.score(Features_test, tags_test))
#print("Accuracy 5 (500):", classifier5.score(Features_test, tags_test))
print("Accuracy 6 (1000):", classifier6.score(Features_test, tags_test))
