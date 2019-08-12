import sys
import nltk
import pyphen
import epitran
import spacy

from inout.dta.corpus import Corpus
from inout.dta.poem import Poem

#from nltk.tokenize import RegexpTokenizer
from nltk import everygrams

from spacy.lang.de.examples import sentences

#tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
dic = pyphen.Pyphen(lang = 'de')
epi = epitran.Epitran('deu-Latn')
nlp = spacy.load('de_core_news_sm', disable=["parser"])

c = Corpus(sys.argv[1])
poems = c.get_poems()


def get_phons(tok_line):
	phon_line = []
	for word in tok_line:
#		phon_line.append(epi.transliterate(word.lower()))
		phon = epi.transliterate(word.lower())
		if word.lower()[0] == 's':
			if word.lower()[1] == 't' or word.lower()[1] == 'p':
				phon = phon.replace('s', 'ʃ', 1)
			#elif word.lower()[1] != 'c':
				#if len(phon) > 2:
					#phon = phon.replace('t͡s', 's', 1)
		phon_line.append(phon)
	return phon_line

def get_everygrams(phon_line):
	a_line = []
	for word in phon_line:
		a_line.append(word[0].lower())
	egrams = everygrams(list(enumerate(a_line)), 1, len(phon_line))
	return egrams

def get_ids(everygrams):
	ids = []
	for egram in everygrams:
		for i, a in egram:
			for k, b in egram:
				if a == b and i != k:
					if (i, k) not in ids and (k, i) not in ids:
						ids.append((i, k))
	return ids

def allit_distance_syllables(tok_line, id1, id2):
	words = []
	for i in range(id1+1, id2):
		words.append(tok_line[i])
	syll_counter = 0
	for word in words:
		syllabified_word = dic.inserted(word)
		syll_counter += len(syllabified_word.split('-'))
	return syll_counter

def create_id_dictionary(tok_line, tag_line, phon_line, id_list):
	id_dic = {}
	for id1, id2 in id_list:
		dist_syllables = allit_distance_syllables(tok_line, id1, id2)
		if dist_syllables < 4:
			if tag_line[id1].pos_ != 'DET' and tag_line[id1].pos_ != 'ADP' and tag_line[id1].pos_ != 'PUNCT' and tag_line[id1].pos_ != 'X' and tag_line[id2].pos_ != 'DET' and tag_line[id2].pos_ != 'ADP' and tag_line[id2].pos_ != 'PUNCT' and tag_line[id2].pos_ != 'X': #and tag_line[id1].text != tag_line[id2].text:
				phon = phon_line[id1][0]
				id_dic.setdefault(phon, []).append(id1)
				id_dic.setdefault(phon, []).append(id2)
	
	for key, value in id_dic.items():
		v = sorted(set(value))
		id_dic[key] = v 
		
	return id_dic

output_file = open('output_parviol_e.txt', 'w')

for poem in poems:
	output_file.write('::: '+poem.get_year()+', '+poem.get_author()+': '+poem.get_title()+' :::'+'\n')
	lines = poem.get_lines()
	set_lines = []
	for line in lines:
		if line not in set_lines:
			set_lines.append(line)
			tok_line = [token.text for token in nlp(line)]
			tag_line = nlp(line)
#			for token in tag_line:
#				output_file.write(token.text+token.pos_+' ')
#			output_file.write('\n')
			print(tok_line)
#			for token in tok_line:
#				syl = dic.inserted(token)
#				output_file.write(syl+'-')
#			output_file.write('\n')
			phon_line = get_phons(tok_line)
#			for phon in phon_line:
#				if len(phon) > 1:
#					if phon[0] == 't' and phon[1] == '͡':
#						print(phon[0]+' '+phon[1]+' : '+phon)
#					else:
#						print('kein t͡s')
#				output_file.write(phon+' ')
#			output_file.write('\n')
			print(phon_line)
			egrams = get_everygrams(phon_line)
			id_list = get_ids(egrams)
			id_dic = create_id_dictionary(tok_line, tag_line, phon_line, id_list)
			if len(id_dic.keys()) == 0:
#				output_file.write('0 ')
#				for token in tok_line:
#					output_file.write(token+' ')
#				output_file.write('\n')
				for phon in phon_line:
					output_file.write(phon+' ')
				output_file.write('\n')
			for key in id_dic:
#				print(key, id_dic[key])
#				output_file.write('X ')
				for x in range(len(tok_line)):
					if x in id_dic[key]:
#						print(tok_line[x].upper())
						output_file.write(tok_line[x].upper()+' ')
					else:
#						print(tok_line[x])
						output_file.write(tok_line[x]+' ')
				output_file.write('\n')
				for phon in phon_line:
					output_file.write(phon+' ')
				output_file.write('\n')
#			output_file.write('\n')
	output_file.write('\n')
output_file.close()
