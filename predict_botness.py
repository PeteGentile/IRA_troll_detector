import praw, pickle, time, string, text_tools
import numpy as np
from praw.models import MoreComments
from cfg import config
from sys import argv
from progress.bar import IncrementalBar


# We're going to need to log in to reddit. To do so, we'll need a number of text
# arguments that I just store in a private config file, which contains a class with
# attributes which store the things we'll need.
cfg = config()

def get_user_comments(username, reddit, verbose = True):
#Given a username, get the comments that user has written. Return them as a list
	comments = []
	
	# There are a number of ways this can go wrong, and as Reddit changes the way the
	# website works, there are bound to be some that pop up which haven't been handled
	# here. This should ensure such situations are handled smoothly.
	try:
		# First get ALL comments by the user.
		user = reddit.redditor(username)
		for c in user.comments.new(limit=None):
			# Then we want to remove any links the user posted in the comment,
			# as well as any text they're quoting from another user. clean_comment does
			# this.
			cc = text_tools.clean_comment(c.body)
			
			# If the remaining comment is empty, don't do anything, otherwise, keep it.
			if len(cc) > 0:
				comments += [cc]
	
	# If the user interrupts, just move to the next user.
	except KeyboardInterrupt:
		pass
	
	# If the comment gathering fails for any other reason, print out the user it failed
	# on, as well as the reason. This facilitates debugging.
	except Exception as e:
		if verbose: 
			print("Barfed on", username)
			print(e)
	
	# Return all the comments as a list.
	return comments


def get_new_users(usernames):
# Once this bot has seen a user, it doesn't have to analyze their comments again, it
# already knows its prediction. If you want to analyze a lot of users at once
# (for example all the users in a thread), you can save a lot of time by only analyzing
# users the bot hasn't seen yet. This makes that possible.
# 
# Note: potential_bots.txt and potential_not_bots.txt are just text files that log
# the results of previous analyses. See the end of the script to see how they're written.

	with open("potential_bots.txt", "r") as f:
			already_analyzed_users = [x for x in f.read().split("\n") if len(x) and\
				not x.startswith("#")]
	with open("potential_not_bots.txt", "r") as f:
			already_analyzed_users += [x for x in f.read().split("\n") if len(x) and\
				not x.startswith("#")]
	
	return [x for x in usernames if not x in already_analyzed_users]


def predict_botness(username, reddit, clf, vectorizer):
# This is the part that actually classifies a given user as a bot or not a bot given a
# previously-trained classifier. This classifier (and vectorizer) are made by 
# model_reddit_comments.py.
	user_comments = get_user_comments(username, reddit)
	new_user_corpus = " ".join(user_comments)
	n_words = len(new_user_corpus.split())
	
	# The classifier does not work well for users who have written fewer than 1000 words,
	# so if that describes the current user, do not bother trying to classify them.
	if n_words < 1000:
		return None
	
	# Otherwise, convert the corpus to a vector and use it to classify the user.
	else:
		features = text_tools.get_text_vectors([new_user_corpus], vectorizer)
		return clf.predict(features)[0] == 1


def get_usernames_from_subreddit(subname):
# One way to use this is to just scrape an entire subreddit. Obviously, you have to stop somewhere, so this takes the users who comment on the current top 10 posts in that sub.
	
	# Some subreddits are "quarantined", which means you have to explicitly consent to
	# entering the subreddit. This try statement does that.
	try:
		subreddit = reddit.subreddit(subname)
		submissions = subreddit.top("day", limit=10)
	except TypeError:
		subreddit = reddit.subreddit(subname)
		subreddit.quaran.opt_in()
		submissions = subreddit.top("day", limit=10)
	
	names = []	
	for s in submissions:
		# For each post in the top 10, get all the users who have commented on it
		names += [c.author.name for c in s.comments.list() if not type(c) == MoreComments\
			and not c.body.startswith("[") and not c.body == "[deleted]" and not\
			c.author == None and not c.author.name.startswith("Unavai") ]
	
	# Return those users
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
	
	usernames = get_new_users(usernames)[:10]
	
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