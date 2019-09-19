import re, nltk, pickle, string, time
from text_tools import clean_comment, get_text_vectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier

def plot_confusion_matrix(test, pred):
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
    if not min_words:
        return {x[0]:x[1] for x in comment_dict.items() if len(x[1]) >= min_comments}
    else:
        outdict = {}
        for uname, comments in comment_dict.items():
            if sum([len(x.split()) for x in comments]) > min_words:
                outdict[uname] = comments
        return outdict

with open("not_bot_comments_0-150.pkl", "rb") as f:
    not_bot_dict = pickle.load(f)
with open("bot_comments_dict.pkl", "rb") as f:
    bot_reddit_dict = pickle.load(f)

bot_reddit_dict = remove_non_commenting_users(bot_reddit_dict, min_words=40)
not_bot_dict = remove_non_commenting_users(not_bot_dict, min_words=1000)

bot_reddit_corpi = [" ".join(bot_reddit_dict[x]) for x in bot_reddit_dict.keys()]
not_bot_reddit_corpi = [" ".join(not_bot_dict[x]) for x in not_bot_dict.keys()]
print("Got comments from %d bots and %d other users." % (len(not_bot_dict.keys()), len(bot_reddit_dict.keys())))

stime = time.time()
print("Started", time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime()))
features, vectorizer = get_text_vectors(bot_reddit_corpi + not_bot_reddit_corpi, vectorizer=TfidfVectorizer())
labels = np.asarray([1]*len(bot_reddit_corpi) + [0]*len(not_bot_reddit_corpi))
print("Getting vectors took %0.1f minutes." % ((time.time()-stime)/60))


kf = KFold(n_splits=6, shuffle=True, random_state=90)
correct_probs = []
incorrect_probs = []
accs = []
preciss = []
recalls = []
i=0

stime = time.time()
print("Started fitting at", time.strftime("%a, %d %b %Y %H:%M:%S %Z\n", time.localtime()))

for train_indices, test_indices in kf.split(features):
    features_train = features[train_indices]
    features_test = features[test_indices]
    labels_train = labels[train_indices]
    labels_test = labels[test_indices]
    clf = GradientBoostingClassifier(learning_rate=0.33, n_estimators=500)
    clf.fit(features_train.toarray(), labels_train)
    pred = clf.predict(features_test.toarray())
    pred_probs = clf.predict_proba(features_test.toarray())
    error_inds = (labels_test==pred)
    correct_probs += [x[1] for x in pred_probs[~error_inds]]
    incorrect_probs += [x[1] for x in pred_probs[error_inds]]
    acc = 100*accuracy_score(labels_test, pred)
    prec = 100*precision_score(labels_test, pred)
    recall = 100*recall_score(labels_test, pred)
    accs.append(acc)
    preciss.append(prec)
    recalls.append(recall)
    print("Number of bots in training set:         ", sum(labels_train))
    print("Number of regular users in training set:", len(labels_train)-sum(labels_train))
    print("Ratio of bots to regular users:          %0.1f" % (sum(labels_train)/(len(labels_train)-sum(labels_train))))
    print("Performance metrics for fold %d:" % (i))
    print("Accuracy:  %0.1f" % (acc))
    print("Precision: %0.1f" % (prec))
    print("Recall:    %0.1f" % (recall))
    print()
    i += 1

print("Average accuracy:  %0.1f" % (sum(accs)/len(accs)))
print("Average precision: %0.1f" % (sum(preciss)/len(preciss)))
print("Average recall:    %0.1f" % (sum(recalls)/len(recalls)))
print("Time to completion: %0.1f min" % ((time.time()-stime)/60))
plot_confusion_matrix(labels_test, pred)

clf.fit(features, labels)

with open("not_bot_comments_all.pkl", "rb") as f:
    all_not_bot_dict = pickle.load(f)

all_not_bot_dict = remove_non_commenting_users(all_not_bot_dict, min_words = 1000)
all_not_bot_reddit_corpi = [" ".join(all_not_bot_dict[x]) for x in all_not_bot_dict.keys()]
len(all_not_bot_reddit_corpi)
stime = time.time()
print("Started geting vectors at", time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime()))
normal_features = get_text_vectors(all_not_bot_reddit_corpi, vectorizer=vectorizer)
normal_labels = np.asarray([0]*len(all_not_bot_reddit_corpi))
print("Getting vectors took %0.1f minutes." % ((time.time()-stime)/60))

pred = clf.predict(normal_features)
pred_probs = clf.predict_proba(normal_features)
print("Incorrectly classified %0.1f percent of innocent users as bots." % (100*sum(pred)/len(pred)))


#with open("final_classifier_vectorizer.pkl", "wb") as f:
#    pickle.dump([clf, vectorizer], f)

clf.predict(features[0].toarray())[0]
