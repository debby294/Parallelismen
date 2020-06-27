import sys
import nltk
import pyphen
import epitran
import spacy

from inout.dta.corpus import Corpus
from inout.dta.poem import Poem

from nltk import everygrams

from spacy.lang.de.examples import sentences

dic = pyphen.Pyphen(lang = 'de')
epi = epitran.Epitran('deu-Latn')
nlp = spacy.load('de_core_news_sm', disable=["parser"])

c = Corpus(sys.argv[1])
poems = c.get_poems()

output_file = open('output_parviol_a_NEU.txt', 'w')

def get_phons(tok_line):
	phon_line = []
	for word in tok_line:
		phon = epi.transliterate(word.lower())
		if word.lower()[0] == 's':
			if word.lower()[1] == 't' or word.lower()[1] == 'p':
				phon = phon.replace('s', 'ʃ', 1)
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
	punct = ',','.',':','-',"'",'"',';'
	for i in range(id1+1, id2):
		if tok_line[i] not in punct:
			words.append(tok_line[i])
	syll_counter = 0
	for word in words:
		syllabified_word = dic.inserted(word)
		syll_counter += len(syllabified_word.split('-'))
	return syll_counter

def create_id_dictionary(tok_line, tag_line, phon_line, id_list):
	id_dic = {}
	id_dic_new = {}
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
	for key, value in id_dic.items():
		num = 1
		for i in range(len(value)):
			phon = phon_line[value[i]][0]
			id_dic_new.setdefault(phon+str(num), []).append(value[i])
			if i + 1 < len(value):
				if allit_distance_syllables(tok_line, value[i], value[i+1]) < 4:
					id_dic_new.setdefault(phon+str(num), []).append(value[i+1])
				else:
					num += 1
					id_dic_new.setdefault(phon+str(num), []).append(value[i+1])
	for key, value in id_dic_new.items():
		v = sorted(set(value))
		id_dic_new[key] = v
	return id_dic_new

for poem in poems:
	info = ':::'+poem.get_year()+': '+poem.get_author()+', '+poem.get_title()+':::'
	print(info)
	output_file.write(info+'\n')
	anz_allit = 0
	double_lines = poem.get_lines()
	laenge = len(double_lines) / 2
	lines = double_lines[:int(laenge)]
	text = ''
	tok_lines = []
	tag_lines = []
	phon_lines = []
	tok_text = []
	for line in lines:
		text = text+line+' '
		tok_line = [token.text for token in nlp(line)]
		tok_lines.append(tok_line)
		phon_lines.append(get_phons(tok_line))
	for line in tok_lines:
		for word in line:
			tok_text.append(word)
	tag_text = nlp(text)
	phon_text = get_phons(tok_text)
	print(tok_text)

#Alliterationen am Versbeginn mit Ausgabe der Alliterationen:
	output_file.write('Alliterationen am Versbeginn:\n')
	for i in range(len(phon_lines)):
		if i + 1 < len(phon_lines) and i != 0:
			if phon_lines[i][0][0] == phon_lines[i+1][0][0] or phon_lines[i][0][0] == phon_lines[i-1][0][0]:
				if phon_lines[i][0][0] == phon_lines[i+1][0][0] and phon_lines[i][0][0] != phon_lines[i-1][0][0]:
					anz_allit += 1
				for x in range(len(tok_lines[i])):
					if x == 0:
						output_file.write(tok_lines[i][x].upper()+' ')
						print(tok_lines[i][x].upper())
					else:
						output_file.write(tok_lines[i][x]+' ')
						print(tok_lines[i][x])
			else:
				print(tok_lines[i])
				for word in tok_lines[i]:
					output_file.write(word+' ')
		elif i + 1 < len(phon_lines) and i == 0:
			if phon_lines[i][0][0] == phon_lines[i+1][0][0]:
				anz_allit += 1
				for x in range(len(tok_lines[i])):
					if x == 0:
						output_file.write(tok_lines[i][x].upper()+' ')
						print(tok_lines[i][x].upper())
					else:
						output_file.write(tok_lines[i][x]+' ')
						print(tok_lines[i][x])
			else:
				print(tok_lines[i])
				for word in tok_lines[i]:
					output_file.write(word+' ')
		else:
			if phon_lines[i][0][0] == phon_lines[i-1][0][0]:
				for x in range(len(tok_lines[i])):
					if x == 0:
						output_file.write(tok_lines[i][x].upper()+' ')
						print(tok_lines[i][x].upper())
					else:
						output_file.write(tok_lines[i][x]+' ')
						print(tok_lines[i][x])
			else:
				print(tok_lines[i])
				for word in tok_lines[i]:
					output_file.write(word+' ')
		output_file.write('\n')
	print(anz_allit)
	output_file.write('\n')
#Alliterationen am Versbeginn nur zählen:
	#for i in range(len(phon_lines)):
		#if i + 1 < len(phon_lines) and i != 0:
			#if phon_lines[i][0][0] == phon_lines[i+1][0][0] and phon_lines[i][0][0] != phon_lines[i-1][0][0]:
				#anz_allit += 1
		#elif i + 1 < len(phon_lines) and i == 0:
			#if phon_lines[i][0][0] == phon_lines[i+1][0][0]:
				#anz_allit += 1
	#print(anz_allit)

#zeilenübergreifende Alliterationen:
	output_file.write('Zeilenübergreifende Alliterationen:\n')
	egrams = get_everygrams(phon_text)
	id_list = get_ids(egrams)
	id_dic = create_id_dictionary(tok_text, tag_text, phon_text, id_list)
	if len(id_dic.keys()) == 0:
		print(tok_text)
	for key in id_dic:
		anz_allit += 1
		for value in id_dic[key]:
			output_file.write(tok_text[value].upper()+' ')
			if value + 1 < len(tok_text) and value + 1 not in id_dic[key]:
				output_file.write(tok_text[value+1]+' ')
			output_file.write('\n')
		#for x in range(len(tok_text)):
			#if x in id_dic[key]:
				#print(tok_text[x].upper())
				#output_file.write(tok_text[x].upper()+' ')
			#else:
				#output_file.write(tok_text[x]+' ')
		output_file.write('\n')
#zeilenübergreifende Alliterationen nur zählen:
	#egrams = get_everygrams(phon_text)
	#id_list = get_ids(egrams)
	#id_dic = create_id_dictionary(tok_text, tag_text, phon_text, id_list)
	#for key in id_dic:
		#anz_allit += 1

	output_file.write('\n'+'Anzahl Alliterationen: '+str(anz_allit)+'\n'+'\n'+'\n')

output_file.close()
