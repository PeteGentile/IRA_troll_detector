import re, nltk, pickle, string, time
from text_tools import clean_comment, get_text_vectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier

def plot_confusion_matrix(test, pred):
	#Given test labels and predicted labels, plot the confusion matrix, and make it
	#pretty.
	labels = np.asarray(['Not Bot', 'Bot'])
	cm = confusion_matrix(labels[test], labels[pred], labels)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm)
	plt.title('Confusion matrix of the classifier')
	fig.colorbar(cax)
	ax.set_xticklabels([''] + list(labels))
	ax.set_yticklabels([''] + list(labels))
	ax.xaxis.set_ticks_position("bottom")
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.tight_layout()
	plt.show()


def remove_non_commenting_users(comment_dict, min_comments = 0, min_words = 0):
	#Some users haven't written much, or not at all. Remove those users because
	#the classifier won't be able to make a reliable prediction on them.
	#
	#The user can also make a cut based on the min number of comments. Also, note the 
	#input is a dictionary with keys being usernames, and items being the comments
	#from that username.
	
	#If the user hasn't specified a min number of words, cut on number of comments.
	if not min_words:
		return {x[0]:x[1] for x in comment_dict.items() if len(x[1]) >= min_comments}
	
	#Else, sum up all the words in all their comments and cut users who have written
	#a number of words below that specified by the user.
	else:
		outdict = {}
		for uname, comments in comment_dict.items():
			#Get the total number of words written. Keep the user's comments if it
			#makes the cut.
			if sum([len(x.split()) for x in comments]) > min_words:
				outdict[uname] = comments
		return outdict

#Load in comments from bots and from normal users
with open("not_bot_comments_0-150.pkl", "rb") as f:
	not_bot_dict = pickle.load(f)
with open("bot_comments_dict.pkl", "rb") as f:
	bot_reddit_dict = pickle.load(f)

#Remove commenters who haven't commented much
bot_reddit_dict = remove_non_commenting_users(bot_reddit_dict, min_words=40)
not_bot_dict = remove_non_commenting_users(not_bot_dict, min_words=1000)

#Combine comments from users such that each element of the corpi list is the collection
#of all the comments written by a single user.
bot_reddit_corpi = [" ".join(bot_reddit_dict[x]) for x in bot_reddit_dict.keys()]
not_bot_reddit_corpi = [" ".join(not_bot_dict[x]) for x in not_bot_dict.keys()]

#Print info about how balanced the dataset is.
print("Got comments from %d bots and %d other users." % (len(not_bot_dict.keys()), len(bot_reddit_dict.keys())))

#Keep track of how long the process is taking.
stime = time.time()
print("Started", time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime()))

#Vectorize the comments. Also keep the vectorizer to use on new comments.
features, vectorizer = get_text_vectors(bot_reddit_corpi + not_bot_reddit_corpi, vectorizer=TfidfVectorizer())

#Label the data correctly.
labels = np.asarray([1]*len(bot_reddit_corpi) + [0]*len(not_bot_reddit_corpi))
print("Getting vectors took %0.1f minutes." % ((time.time()-stime)/60))

#We'll use K-fold cross validation here, where K is 6. This will allow a good number
#of folds with each fold having a decent amount of data in it
kf = KFold(n_splits=6, shuffle=True, random_state=90)
correct_probs = []
incorrect_probs = []
accs = []
preciss = []
recalls = []
i=0

#Keep track of how long the process is taking.
stime = time.time()
print("Started fitting at", time.strftime("%a, %d %b %Y %H:%M:%S %Z\n", time.localtime()))

#For each fold, train the model and print out performance metrics for it.
#The metric we care about most is precision, because if we're calling someone a Russian
#troll, we want to make sure we're right.
#The learning rate and n_estimators have been tuned externally.
for train_indices, test_indices in kf.split(features):
	#Get features and labels.
	features_train = features[train_indices]
	features_test = features[test_indices]
	labels_train = labels[train_indices]
	labels_test = labels[test_indices]
	
	#Make and fit the classifier.
	clf = GradientBoostingClassifier(learning_rate=0.33, n_estimators=500)
	clf.fit(features_train.toarray(), labels_train)
	
	#Make predictions.
	pred = clf.predict(features_test.toarray())
	pred_probs = clf.predict_proba(features_test.toarray())
		
	#Calculate and print information about each fold's training set/performance metrics.
	acc = 100*accuracy_score(labels_test, pred)
	prec = 100*precision_score(labels_test, pred)
	recall = 100*recall_score(labels_test, pred)
	accs.append(acc)
	preciss.append(prec)
	recalls.append(recall)
	print("Number of bots in training set:		 ", sum(labels_train))
	print("Number of regular users in training set:", len(labels_train)-sum(labels_train))
	print("Ratio of bots to regular users:		  %0.1f" % (sum(labels_train)/(len(labels_train)-sum(labels_train))))
	print("Performance metrics for fold %d:" % (i))
	print("Accuracy:  %0.1f" % (acc))
	print("Precision: %0.1f" % (prec))
	print("Recall:	%0.1f" % (recall))
	print()
	i += 1

#Print the average performance of the model.
print("Average accuracy:  %0.1f" % (sum(accs)/len(accs)))
print("Average precision: %0.1f" % (sum(preciss)/len(preciss)))
print("Average recall:	%0.1f" % (sum(recalls)/len(recalls)))
print("Time to completion: %0.1f min" % ((time.time()-stime)/60))

#Plot the confusion matrix
plot_confusion_matrix(labels_test, pred)

#Fit on all features/labels for final classifier.
clf.fit(features, labels)

#Now test the classifier out on users it hasn't seen before. Note that *none* of these
#should be bots/trolls.
with open("not_bot_comments_all.pkl", "rb") as f:
	all_not_bot_dict = pickle.load(f)

#Prep the data as before.
all_not_bot_dict = remove_non_commenting_users(all_not_bot_dict, min_words = 1000)
all_not_bot_reddit_corpi = [" ".join(all_not_bot_dict[x]) for x in all_not_bot_dict.keys()]

#Keep track of how long the process is taking.
stime = time.time()
print("Started geting vectors at", time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime()))
normal_features = get_text_vectors(all_not_bot_reddit_corpi, vectorizer=vectorizer)
normal_labels = np.asarray([0]*len(all_not_bot_reddit_corpi))
print("Getting vectors took %0.1f minutes." % ((time.time()-stime)/60))

#Make predictions based on the new users.
pred = clf.predict(normal_features)
pred_probs = clf.predict_proba(normal_features)
print("Incorrectly classified %0.1f percent of innocent users as bots." % (100*sum(pred)/len(pred)))

#Save the classifier and vectorizer.
#with open("final_classifier_vectorizer.pkl", "wb") as f:
#	pickle.dump([clf, vectorizer], f)