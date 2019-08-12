print('import')
import pickle
import random
import sklearn_crfsuite

classifier = sklearn_crfsuite.CRF()

def word2features(sentence, index):
	word = sentence[index][0]
	postag = sentence[index][1]
	features = {
	# übernommen vom DecisionTreeClassifier
		'word': word,
		'is_first': index == 0,
		'is_last': index == len(sentence) - 1,
		'is_capitalized': word[0].upper() == word[0],
		'is_all_caps': word.upper() == word,
		'is_all_lower': word.lower() == word,
		'prefix-1': word[0],
		'prefix-2': word[:2],
		'prefix-3': word[:3],
		'suffix-1': word[-1],
		'suffix-2': word[-2:],
		'suffix-3': word[-3:],
#		'prev_word': '' if index == 0 else sentence[index - 1],
#		'next_word': '' if index == len(sentence) - 1 else sentence[index + 1], #funktionieren bei diesem Classifier so nicht, da das Ergebnis hier tuples sind, mit denen der Classifier nichts anfangen kann
# stattdessen:
		'prev_word': '' if index == 0 else sentence[index-1][0],
		'prev_tag': '' if index == 0 else sentence[index-1][1],
		'next_word': '' if index == len(sentence)-1 else sentence[index+1][0],
		'next_tag': '' if index == len(sentence)-1 else sentence[index+1][1],
		'has_hyphen': '-' in word,
		'is_numeric': word.isdigit(),
		'capitals_inside': word[1:].lower() != word[1:]
	}
	return features

def sent2features(sentence):
	return [word2features(sentence, i) for i in range(len(sentence))]

def sent2labels(sentence):
	return [w[1] for w in sentence]

print('loading corpus')
with open ('dta_komplett_tcf-full-ohne_lyrik-1000000sents', 'rb') as f:
	train_sents = pickle.load(f)

with open ('dta_komplett_tcf-full-lyrik-1000000sents', 'rb') as f:
	sents = pickle.load(f)

print('preparing corpus')
random.shuffle(sents)
random.shuffle(train_sents)

cutoff = int(0.8 * len(sents))
#train_sents = sents[:cutoff]
test_sents = sents[cutoff:]

Features = [sent2features(s) for s in train_sents]
tags = [sent2labels(s) for s in train_sents]

print('training')
classifier.fit(Features, tags)

print('testing')
Features_test = [sent2features(s) for s in test_sents]
tags_test = [sent2labels(s) for s in test_sents]
print('Accuracy: ', classifier.score(Features_test, tags_test))
