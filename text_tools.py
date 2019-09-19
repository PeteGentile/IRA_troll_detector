import praw, nltk, string
import numpy as np
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = list(set(stopwords.words('english'))) + ["the"]

def get_user_comments(username, reddit, verbose = True):
	comments = []
	
	pct = 0
	try:
		user = reddit.redditor(username)
		for c in user.comments.new(limit=None):
			cc = clean_comment(c.body)
			if len(cc) > 0:
				comments += [cc]
	except KeyboardInterrupt:
		pass
	except Exception as e:
		if verbose: 
			print("Barfed on", username)
			print(e)
	return comments


def clean_comment(comment_string):
	lines = comment_string.split("\n")
	clean_lines = []
	for line in lines:
		words = []
		if line.startswith(">"):
			pass
		else:
			for w in line.split():
				if w.startswith(("[", "http", "www", "@", "#")):
					pass
				else:
					words.append(w.replace("\n\n"," ").replace("°",""))
			clean_lines.append(" ".join(words))
		
	return "\n".join(clean_lines)


def get_text_vectors(comments, vectorizer):
	lemmas = transform_and_lemmatize(comments)
	if hasattr(vectorizer, "vocabulary_"):
		vector = vectorizer.transform(lemmas)
		return vector
	else:
		vector = vectorizer.fit_transform(lemmas)
		return vector, vectorizer


def transform_and_lemmatize(comments):
	stemmer = WordNetLemmatizer()
	output = []
	for c in comments:
		c = remove_punctuation(c)
		tokens = word_tokenize(c)
		filtered_tokens = remove_stop_words(tokens)
		lemmas = lemmatize(filtered_tokens, stemmer)
		output.append(" ".join(lemmas))
	return output


def remove_punctuation(comment):
	for p in string.punctuation:
		comment = comment.replace(p," ")
	return comment


def remove_stop_words(word_list):
	return [word for word in word_list if not word in stop_words]


def lemmatize(comment, stemmer):
	parts_of_speech = [get_wordnet_pos(word) for word in comment]
	output = [stemmer.lemmatize(word, pos) for word, pos in zip(comment, parts_of_speech)]
	return output


def get_wordnet_pos(word):
	"""Map POS tag to first character lemmatize() accepts"""
	tag = nltk.pos_tag([word])[0][1][0].upper()
	tag_dict = {"J": wordnet.ADJ,
				"N": wordnet.NOUN,
				"V": wordnet.VERB,
				"R": wordnet.ADV}

	return tag_dict.get(tag, wordnet.NOUN)

