import praw, nltk, string
import numpy as np
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = list(set(stopwords.words('english'))) + ["the"]

def get_user_comments(username, reddit, verbose = True):
	#This function gets the comments from a given user
	comments = []
	
	#We want to handle unexpected circumstances elegantly, so we'll use a try statement
	try:
		#Get the user
		user = reddit.redditor(username)
		
		#Get all their comments, and for each one, take out text they've quoted from 
		#other users, links they posted, hashtage, etc.
		for c in user.comments.new(limit=None):
			cc = clean_comment(c.body)
			
			#If there's anything left, keep it
			if len(cc) > 0:
				comments += [cc]
	except KeyboardInterrupt:
		pass
		
	#If there's an exception, let the user know about it.
	except Exception as e:
		if verbose: 
			print("Barfed on", username)
			print(e)
	
	#Return all the comments
	return comments


def clean_comment(comment_string):
	#Given a user's comment, take out text they've quoted from other users,
	#links they posted, hashtage, etc.
	lines = comment_string.split("\n")
	clean_lines = []
	for line in lines:
		words = []
		
		#When a user quotes another user, the line starts with a >, so take out
		#those lines
		if line.startswith(">"):
			pass
		else:
			#Now clean out all the other stuff stated above.
			for w in line.split():
				if w.startswith(("[", "http", "www", "@", "#")):
					pass
				else:
					words.append(w.replace("\n\n"," ").replace("°",""))
			#Put the comment back together and append it to the total comment list.
			clean_lines.append(" ".join(words))
	#Return the cleaned comment
	return "\n".join(clean_lines)


def get_text_vectors(comments, vectorizer):
	#Transform, lemmatize, and use the given vectorizer to get vectors of the text.
	lemmas = transform_and_lemmatize(comments)
	
	#If the user specifies a vectorizer that's already been fit, just vectorize the
	#given comments
	if hasattr(vectorizer, "vocabulary_"):
		vector = vectorizer.transform(lemmas)
		return vector
	#If the vectorizer hasn't already been fit, fit it, then vectorize the given comments
	else:
		vector = vectorizer.fit_transform(lemmas)
		return vector, vectorizer


def transform_and_lemmatize(comments):
	#Use the WordNetLemmatizer to lemmatize the given comments
	stemmer = WordNetLemmatizer()
	output = []
	
	#First, we'll want to remove punctuation and stop words. Then we'll lemmatize
	#the comment.
	for c in comments:
		c = remove_punctuation(c)
		tokens = word_tokenize(c)
		filtered_tokens = remove_stop_words(tokens)
		lemmas = lemmatize(filtered_tokens, stemmer)
		output.append(" ".join(lemmas))
	#Return the lemmatized comment
	return output


def remove_punctuation(comment):
	#Remove all the punctuation from a comment.
	for p in string.punctuation:
		comment = comment.replace(p," ")
	return comment


def remove_stop_words(word_list):
	#Given a list of words, return only those that aren't in the list of stop words
	#defined in the NLTK corpus.
	return [word for word in word_list if not word in stop_words]


def lemmatize(comment, stemmer):
	#Get the part of speech (from wordnet), then lemmatize the word.
	parts_of_speech = [get_wordnet_pos(word) for word in comment]
	output = [stemmer.lemmatize(word, pos) for word, pos in zip(comment, parts_of_speech)]
	return output


def get_wordnet_pos(word):
	#Map POS tag to first character lemmatize() accepts. These letters are predefined
	#by wordnet.
	tag = nltk.pos_tag([word])[0][1][0].upper()
	tag_dict = {"J": wordnet.ADJ,
				"N": wordnet.NOUN,
				"V": wordnet.VERB,
				"R": wordnet.ADV}
	#Return the dict.
	return tag_dict.get(tag, wordnet.NOUN)

