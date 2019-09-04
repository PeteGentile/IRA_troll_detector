
# coding: utf-8

# In[26]:


import praw, re, nltk, pickle, string, time
import pandas as pd
import numpy as np
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from praw.models import MoreComments
from cfg import config
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

stop_words = list(set(stopwords.words('english'))) + ["the"]
cfg = config()


# In[11]:


reddit = praw.Reddit(client_id = cfg.client_id,                     username = cfg.username,                      password = cfg.password,                     client_secret = cfg.secret,                     user_agent = cfg.agent)


# In[12]:


def get_user_comments(usernames, get_other_users = True, verbose = True):
    comments = {}
    threads = []
    not_bot_users = []
    
    stime = time.time()
    n_unames = len(usernames)
    pct = 0
    if verbose: print("Started", time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime()))
    for i, u in enumerate(usernames):
        try:
            user = reddit.redditor(u)
            comments[u] = []
            for c in user.comments.new(limit=None):
                cc = clean_comment(c.body)
                if len(cc) > 0:
                    comments[u] += [cc]
                if get_other_users:
                    thread = c.submission
                    if not thread in threads:
                        threads.append(thread)
                        not_bot_users += [list(set([x.author.name for x in thread.comments.list()                                                    if not type(x) == MoreComments and x.author and                                                    not x.author.name in cfg.bot_names]))]
                
            if 100*i/len(usernames) > pct:
                pct += 2.5
                print("Finished %0.1f percent of users in %0.1f minutes." % (100*i/len(usernames), (time.time()-stime)/60))
        except KeyboardInterrupt:
            break
        except Exception as e:
            if verbose: 
                print("Barfed on", u)
                print(e)

    if verbose: print('Finished in %0.1f minutes' % ((time.time()-stime)/60))
    #return transform_and_lemmatize(comments), threads
    return comments, threads, not_bot_users


# In[13]:


def get_most_common_users(ulist):
    all_users = [name for thread_authors in ulist for name in thread_authors]
    unique_users = list(set(all_users))
    uname_incidences = [all_users.count(x) for x in unique_users]
    sorted_unames = [name for _,name in sorted(zip(uname_incidences, unique_users), reverse=True)]
    print(len(unique_users), len(uname_incidences))
    return sorted_unames


# In[14]:


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


# In[15]:


def get_text_vectors(comments, vectorizer = None, remove_stops = True, remove_punc = True, fit = True):
    if vectorizer == None:
        vectorizer = CountVectorizer()
    lemmas = transform_and_lemmatize(comments, remove_stops, remove_punc)
    if fit:
        vector = vectorizer.fit_transform(lemmas)
    else:
        vector = vectorizer.transform(lemmas)
    return vector, vectorizer


# In[16]:


def transform_and_lemmatize(comments, remove_stops = True, remove_punc = True):
    stemmer = WordNetLemmatizer()
    output = []
    for c in comments:
        if remove_punc:
            c = remove_punctuation(c)
        tokens = word_tokenize(c)
        if remove_stops:
            filtered_tokens = remove_stop_words(tokens)
        else:
            filtered_tokens = tokens
        lemmas = lemmatize(filtered_tokens, stemmer)
        output.append(" ".join(lemmas))
    return output


# In[17]:


def remove_punctuation(comment):
    for p in string.punctuation:
        comment = comment.replace(p," ")
    return comment


# In[18]:


def remove_stop_words(word_list):
    return [word for word in word_list if not word in stop_words]


# In[19]:


def lemmatize(comment, stemmer):
    parts_of_speech = [get_wordnet_pos(word) for word in comment]
    output = [stemmer.lemmatize(word, pos) for word, pos in zip(comment, parts_of_speech)]
    return output


# In[20]:


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# In[21]:


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


# In[22]:


def remove_non_commenting_users(comment_dict, min_comments = 0, min_words = 0):
    if not min_words:
        return {x[0]:x[1] for x in comment_dict.items() if len(x[1]) >= min_comments}
    else:
        outdict = {}
        for uname, comments in comment_dict.items():
            if sum([len(x.split()) for x in comments]) > min_words:
                outdict[uname] = comments
        return outdict


# In[46]:


with open("not_bot_comments_0-150.pkl", "rb") as f:
    not_bot_dict = pickle.load(f)
with open("bot_comments_dict.pkl", "rb") as f:
    bot_reddit_dict = pickle.load(f)

bot_reddit_dict = remove_non_commenting_users(bot_reddit_dict, min_words=40)
not_bot_dict = remove_non_commenting_users(not_bot_dict, min_words=1000)

bot_reddit_corpi = [" ".join(bot_reddit_dict[x]) for x in bot_reddit_dict.keys()]
not_bot_reddit_corpi = [" ".join(not_bot_dict[x]) for x in not_bot_dict.keys()]
print("Got comments from %d bots and %d other users." % (len(not_bot_dict.keys()), len(bot_reddit_dict.keys())))


# In[47]:


stime = time.time()
print("Started", time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime()))
features, vectorizer = get_text_vectors(bot_reddit_corpi + not_bot_reddit_corpi, vectorizer=TfidfVectorizer(), remove_stops = True)
labels = np.asarray([1]*len(bot_reddit_corpi) + [0]*len(not_bot_reddit_corpi))
print("Getting vectors took %0.1f minutes." % ((time.time()-stime)/60))


# In[48]:


kf = KFold(n_splits=6, shuffle=True, random_state=90)
correct_probs = []
incorrect_probs = []
accs = []
preciss = []
recalls = []
i=0

stime = time.time()
print("Started", time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime()))

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


# In[49]:


clf.fit(features, labels)


# In[50]:


with open("not_bot_comments_all.pkl", "rb") as f:
    all_not_bot_dict = pickle.load(f)

all_not_bot_dict = remove_non_commenting_users(all_not_bot_dict, min_words = 1000)
all_not_bot_reddit_corpi = [" ".join(all_not_bot_dict[x]) for x in all_not_bot_dict.keys()]
len(all_not_bot_reddit_corpi)
stime = time.time()
print("Started geting vectors at", time.strftime("%a, %d %b %Y %H:%M:%S %Z", time.localtime()))
normal_features, normal_vectorizer = get_text_vectors(all_not_bot_reddit_corpi, vectorizer=vectorizer, fit=False, remove_stops = True)
normal_labels = np.asarray([0]*len(all_not_bot_reddit_corpi))
print("Getting vectors took %0.1f minutes." % ((time.time()-stime)/60))


# In[51]:


pred = clf.predict(normal_features)
pred_probs = clf.predict_proba(normal_features)
print("Incorrectly classified %0.1f percent of innocent users as bots." % (100*sum(pred)/len(pred)))


# In[52]:


sum(pred), len(pred)


# In[38]:


len(bot_reddit_corpi)


# In[54]:


with open("final_classifier_vectorizer.pkl", "wb") as f:
    pickle.dump([clf, vectorizer], f)


# In[64]:


clf.predict(features[0].toarray())[0]

