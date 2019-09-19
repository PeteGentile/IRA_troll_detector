import praw, pickle, time, string, text_tools
import numpy as np
from praw.models import MoreComments
from cfg import config
from sys import argv
from progress.bar import IncrementalBar

cfg = config()

def get_new_users(usernames):
	with open("potential_bots.txt", "r") as f:
			already_analyzed_users = [x for x in f.read().split("\n") if len(x) and\
				not x.startswith("#")]
	with open("potential_not_bots.txt", "r") as f:
			already_analyzed_users += [x for x in f.read().split("\n") if len(x) and\
				not x.startswith("#")]
	
	return [x for x in usernames if not x in already_analyzed_users]


def predict_botness(username, reddit, clf, vectorizer):
	user_comments = get_user_comments(username, reddit)
	new_user_corpus = " ".join(user_comments)
	n_words = len(new_user_corpus.split())
	#print(username, "has written %d words." % n_words)
	if n_words < 1000:
		#print("WARNING!", username, "has only written %d words." % n_words)
		return None
	else:
		features = text_tools.get_text_vectors([new_user_corpus], vectorizer)
		return clf.predict(features)[0] == 1


def get_usernames_from_subreddit(subname):
	try:
		subreddit = reddit.subreddit(subname)
	except TypeError:
		subreddit = reddit.subreddit(subname)
		subreddit.quaran.opt_in()
	submissions = subreddit.top(limit=10)
	names = []
	for s in submissions:
		sub = s
		names += [c.author.name for c in s.comments.list() if not type(c) == MoreComments\
			and not c.body.startswith("[") and not c.body == "[deleted]" and not\
			c.author == None and not c.author.name.startswith("Unavai") ]
	
	return names



if __name__  == '__main__':
	modelfile = "final_classifier_vectorizer.pkl"
	dont_analyze = ["AutoModerator", "autotldr"]
	reddit = praw.Reddit(client_id = cfg.client_id, username = cfg.username,\
		password = cfg.password, client_secret = cfg.secret, user_agent = cfg.agent)
	
	for i, arg in enumerate(argv):
		if arg == "-u":
			usernames = set(argv[i+1:])
		elif arg == "-f":
			with open(argv[i+1], "r") as f:
				usernames = [x for x in f.read().split("\n") if len(x) and\
					not x.startswith("#")][0].split()
		elif arg == "-m":
			modelfile = argv[i+1]
		elif arg == "-sub":
			subname = argv[i+1]
			usernames = get_usernames_from_subreddit(subname)
	
	with open(modelfile, "rb") as f:
		clf, vectorizer = pickle.load(f)
	
	usernames = get_new_users(usernames)
	
	bots = []
	not_bots = []
	insufficient_comments = []
	
	start_str = time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime())
	print()
	print("/r/" + subname + " search started at", start_str)
	bar = IncrementalBar("Analyzing Users", max = len(usernames), suffix = '%(percent)d%% [%(elapsed_td)s / %(eta_td)s]')
	for username in set(usernames):
		if not username in dont_analyze:
			botness = predict_botness(username, reddit, clf, vectorizer)
	
			if botness == None:
				#print(username, "has not commented enough to create a reliable prediction.")
				insufficient_comments.append(username)
			elif botness:
				#print(username, "is likely a bot.")
				bots.append(username)
			elif not botness:
				#print(username, "is likely not a bot.")
				not_bots.append(username)
		else:
			pass
		
		bar.next()
	
	bar.finish()
	
	if len(bots):
		print()
		print("The model predicts the following names to BE BOTS:")
		for b in bots:
			print(b)
		
		with open("potential_bots.txt", "a+") as f:
			f.write("\n#Search at " + start_str + "\n")
			for b in bots:
				f.write(b + "\n")
		
	else:
		print()
		print("The model does not predict any of the supplied usernames to be bots.")
	
	if len(not_bots):
		print()
		print("The model predicts the following names to NOT BE BOTS:")
		for nb in not_bots:
			print(nb)
		
		with open("potential_not_bots.txt", "a+") as f:
			f.write("\n#Search at " + start_str + "\n")
			for nb in not_bots:
				f.write(nb + "\n")

	
	if len(insufficient_comments):
		print()
		print("The following users have not written enough for the model to predict their botness (min 1000 words written):")
		for ic in insufficient_comments:
			print(ic)