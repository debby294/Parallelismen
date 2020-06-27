import sys
import pyphen
import epitran
import nltk
import joblib

from nltk.tokenize import RegexpTokenizer

from inout.dta.corpus import Corpus
from inout.dta.poem import Poem

# sys.argv[1] Corpus path : ../../resources/Reim_Korpora/A_E_Parviol_Korpus/A_Parviol_Korpus
# sys.argv[2] meter model path : ./meter/meter.model.joblib

pyp = pyphen.Pyphen(lang = 'de')
epi = epitran.Epitran('deu-Latn')
#tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')

c = Corpus(sys.argv[1])
poems = c.get_poems()

###für das meter model###
meter_model = sys.argv[2]
clf = joblib.load(meter_model)
tokenizer = RegexpTokenizer(r'\w+')

def word2features(sentence, index):
        word = sentence[index]
        #print(word, len(word), "	", index)
        #postag = sentence[index][1]
        features = {
        # uebernommen vom DecisionTreeClassifier
                'word': word,
                'position_in_sentence': index,
                'rel_position_in_sentence': index / len(sentence),
                'is_first': index == 0,
                'is_last': index == len(sentence) - 1,
                'is_capitalized': word[0].upper() == word[0],
                'next_capitalized': '' if index == len(sentence) -1 else sentence[index+1].upper() == sentence[index+1],
                'last_capitalized': '' if index == 0 else sentence[index-1].upper() == sentence[index-1],
                'is_all_caps': word.upper() == word,
                'is_all_lower': word.lower() == word,
                'prefix-1-low': word[0].lower(),
                'prefix-1': word[0],
                'prefix-2': word[:2],
                'prefix-3': word[:3],
                'prefix-4': word[:4],
                'suffix-1': word[-1],
                'suffix-2': word[-2:],
                'suffix-3': word[-3:],
                'suffix-4': word[-4:],
                'prev_word': '' if index == 0 else sentence[index-1],
                'prev_prev_word': '' if index == 0 or index == 1 else sentence[index-2],
                'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
                'next_next_word': '' if index == len(sentence) - 1 or index == len(sentence) -2  else sentence[index + 2],
                #'prev_tag': '' if index == 0 else sentence[index-1][1],
                #'next_tag': '' if index == len(sentence)-1 else sentence[index+1][1],
                'has_hyphen': '-' in word,
                'is_numeric': word.isdigit(),
                'capitals_inside': word[1:].lower() != word[1:]
        }
        return features


def sent2features(sentence):
        return [word2features(sentence, i) for i in range(len(sentence))]

def analyze_meter(meter_model, string_line):
	tokenized_line = tokenizer.tokenize(string_line)
	syllable_line = []
	for token in tokenized_line:
		hyphenated = dic.inserted(token)
		syllables = hyphenated.split('-')
		for syllable in syllables:
			syllable_line.append((syllable.strip(), ''))
	#print(syllable_line)

	line_features = sent2features(syllable_line)
	#print('Load Meter Model')
	clf = joblib.load(meter_model)
	#print('Predict Meter')
	pred = clf.predict([line_features])
	return pred

###_###


output_file = open('output_try6.txt', 'w')
output_file.write(':::ASSONANZEN PARVIOL:::\n\n')

def get_phons(tok_line):
	phon_line = []
	#for word in tok_line:
	#	phon_line.append(epi.transliterate(word.lower()))
	for word in tok_line:
		phon = epi.transliterate(word.lower())
		if 'iə' in phon:
			phon = phon.replace('iə', 'ï')
		if 'áë' in phon:
			phon = phon.replace('áë', 'á')
		phon_line.append(phon)
	return phon_line

def syllable_distance(syl_line, id1, id2):
	words = []
	punct = ',','.',':','-',"'",'"',';'
	for i in range(id1+1, id2):
		if tok_line[i] not in punct:
			words.append(syl_line[i])
	return len(words)


for poem in poems:
	nr = 0
	info = ':::'+poem.get_year()+': '+poem.get_author()+', '+poem.get_title()+':::'
	print(info)
	output_file.write(info+'\n')
	anz_ass = 0
	double_lines = poem.get_lines()
	laenge = len(double_lines) / 2
	lines = double_lines[:int(laenge)]
	for line in lines:
		nr +=1
		anz_line = 0
		phon_dic = {}
		phon_dic.setdefault('a', [])
		phon_dic.setdefault('e', [])
		phon_dic.setdefault('i', [])
		phon_dic.setdefault('o', [])
		phon_dic.setdefault('u', [])

		phon_dic.setdefault('ä', [])
		phon_dic.setdefault('ë', [])
		phon_dic.setdefault('ï', [])
		phon_dic.setdefault('ö', [])
		phon_dic.setdefault('ü', [])

		phon_dic.setdefault('y', [])
		phon_dic.setdefault('ø', [])
		phon_dic.setdefault('ə', [])
		phon_dic.setdefault('á', [])
		phon_dic.setdefault('â', [])
		phon_dic.setdefault('ᴔ', [])
		phon_dic.setdefault('æ', [])
		tok_line = tokenizer.tokenize(line)
		syl_line = []
		for word in tok_line:
			syls = pyp.inserted(word)
			for syl in syls.split('-'):
				syl_line.append(syl)
		phon_tok_line = get_phons(tok_line)
		phon_syl_line = get_phons(syl_line)
		features_line = sent2features(syl_line)
		meter_lines = clf.predict([features_line])
		for sub in meter_lines:
			meter_line = sub
		print(meter_line)

		for key in phon_dic:
			for i in range(len(phon_syl_line)):
				if key in phon_syl_line[i]:
					phon_dic[key].append(i)

##einfache Assonanzen in der ganzen Zeile:
		#for key in phon_dic:
			#if len(phon_dic[key]) > 1:
				#anz_ass += 1
				#anz_line += 1
				#for i in range(len(syl_line)):
					#if i in phon_dic[key]:
						#output_file.write(syl_line[i].upper()+' ')
					#else:
						#output_file.write(syl_line[i]+' ')
				#output_file.write('\n')
		#if anz_line == 0:
			#output_file.write('XXX ')
			#for syl in syl_line:
				#output_file.write(syl+' ')
			#output_file.write('\n')

			#print(phon_dic)
			#print()
	#output_file.write('\n'+'Anzahl Assonanzen: '+str(anz_ass)+'\n'+'\n'+'\n')
	
	##einfache Assonanzen mit max 3 Silben dazwischen:
		#phon_dic_new = {}
		#print(info, "\t", nr)
		#for key, value in phon_dic.items():
			#num = 1
			#prev_value = -1
			#for i in range(len(value)):
				#phon_dic_new.setdefault(key+str(num), []).append(value[i])
				#if i + 1 < len(value):
					#if (value[i+1] - value[i]) < 5:
						#phon_dic_new.setdefault(key+str(num), []).append(value[i+1])
					#else:
						#num += 1
						#phon_dic_new.setdefault(key+str(num), []).append(value[i+1])
		#for key, value in phon_dic_new.items():
			#v = sorted(set(value))
			#phon_dic_new[key] = v
		
		#for key in phon_dic_new:
			#if len(phon_dic_new[key]) > 1:
				#anz_ass += 1
				#anz_line += 1
				#for i in range(len(syl_line)):
					#if i in phon_dic_new[key]:
						#output_file.write(syl_line[i].upper()+' ')
					#else:
						#output_file.write(syl_line[i]+' ')
				#output_file.write('\n')
		#if anz_line == 0:
			#for syl in syl_line:
				#output_file.write(syl+' ')
			#output_file.write('\n')
	#output_file.write('\n'+'Anzahl Assonanzen: '+str(anz_ass)+'\n'+'\n'+'\n')
	
	#betonte Assonanzen mit max 1 betonten Silbe dazwischen:
		phon_dic_dis = {}
		print(info, "\t", nr)
		for key, value in phon_dic.items():
			num = 1
			for i in range(len(value)):
				phon_dic_dis.setdefault(key+str(num), []).append(value[i])
				if i + 1 < len(value):
					if meter_line[value[i]:value[i+1]].count('+') < 2 :
						phon_dic_dis.setdefault(key+str(num), []).append(value[i+1])
					else:
						num += 1
						phon_dic_dis.setdefault(key+str(num), []).append(value[i+1])
		for key, value in phon_dic_dis.items():
			v = sorted(set(value))
			phon_dic_dis[key] = v
		phon_dic_str = {}
		for key, value in phon_dic_dis.items():
			for i in range(len(value)):
				if meter_line[value[i]] == '+':
					phon_dic_str.setdefault(key, []).append(value[i])
		
		#for key, value in phon_dic.items():
			#num = 1
			#for i in range(len(value)):
				#if meter_line[value[i]] == '+':
					#phon_dic_dis.setdefault(key-str(num), []).append(value[i])
					#if i +1 < len(value):
						#if meter_line[value[i]:value[i+1]].count('+') < 2:
							#if meter_line[value[i+1]] == '+':
								#phon_dic_dis.setdefault(key+str(num), [].append(value[i+1]))
						#else:
							#if meter_line[value[i+1]] == '+':
								#num += 1
								#phon_dic_dis.setdefault(key+str(num), []).append(value[i+1])
		for key, value in phon_dic_dis.items():
			v = sorted(set(value))
			phon_dic_dis[key] = v
		
		for key in phon_dic_str:
			if len(phon_dic_str[key]) > 1:
				anz_ass += 1
				anz_line += 1
				for i in range(len(syl_line)):
					if i in phon_dic_str[key]:
						output_file.write(syl_line[i].upper()+' ')
						print(syl_line[i], ' ', i)
					else:
						output_file.write(syl_line[i]+' ')
				output_file.write('\n')
		if anz_line == 0:
			for syl in syl_line:
				output_file.write(syl+' ')
			output_file.write('\n')
		for meter in meter_line:
			output_file.write(meter+' ')
		output_file.write('\n')
		for syl in phon_syl_line:
			output_file.write(syl+' ')
		output_file.write('\n\n')
	output_file.write('\n'+'Anzahl Assonanzen: '+str(anz_ass)+'\n'+'\n'+'\n')

output_file.close()
