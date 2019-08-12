print('importing')
import pickle
import random

from inout.dta.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger

print('loading')
with open('1000_sents', 'rb') as f:
	sents = pickle.load(f)

print('preparing')
random.shuffle(sents)

cutoff = int(0.9 * len(sents))
train_sents = sents[:cutoff]
test_sents = sents[cutoff:]

print('training')
tagger = ClassifierBasedGermanTagger(train = train_sents)

print('evaluating')
accuracy = tagger.evaluate(test_sents)
print('accuracy: ', accuracy)
