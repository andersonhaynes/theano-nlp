
from data_utils import get_file
import string
import cPickle

def text_to_word_sequence(text):
    text = text.lower()
    text = text.translate(string.maketrans("",""), string.punctuation)
    return text.split()


def make_reuters_dataset(path='temp/reuters21578/'):
    import os
    import re

    min_samples_per_topic = 15

    wire_topics = []
    topic_counts = {}
    wire_bodies = []

    for fname in os.listdir(path):
        if 'sgm' in fname:
            s = open(path + fname).read()
            tag = '<TOPICS>'
            while tag in s:
                s = s[s.find(tag)+len(tag):]
                topics = s[:s.find('</')]
                
                if topics and not '</D><D>' in topics:
                    topic = topics.replace('<D>', '').replace('</D>', '')
                    wire_topics.append(topic)

                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

                else:
                    continue

                bodytag = '<BODY>'
                body = s[s.find(bodytag)+len(bodytag):]
                body = body[:body.find('</')]
                wire_bodies.append(body)

    items = topic_counts.items()
    items.sort(key = lambda x: x[1])
    kept_topics = set()
    for x in items:
        print x[0] + ': ' + str(x[1])
        if x[1] >= min_samples_per_topic:
            kept_topics.add(x[0])
    print '-'
    print 'Kept topics:', len(kept_topics)

    sequences = []
    labels = []
    topic_indexes = {}
    word_counts = {}
    for t, b in zip(wire_topics, wire_bodies):
        if t in kept_topics:
            if t not in topic_indexes:
                topic_index = len(topic_indexes)
                topic_indexes[t] = topic_index
            else:
                topic_index = topic_indexes[t]

            labels.append(topic_index)

            word_seq = text_to_word_sequence(b)
            sequences.append(word_seq)
            for w in word_seq:
                word_counts[w] = word_counts.get(w, 0) + 1

    print 'Samples in dataset:', len(labels)

    items = word_counts.items()
    items.sort(key = lambda x: x[1], reverse=True)
    sorted_voc = [x[0] for x in items]
    word_index = dict(zip(sorted_voc, range(len(sorted_voc))))
    print '-'
    print 'Complete vocabulary:', len(word_index)

    print 'Sanity check:'
    for w in ['the', 'in', 'coal', 'pacific', 'blah']:
        print w + ': ' + str(word_index.get(w))

    X = []
    for s in sequences:
        ns = []
        for w in s:
            ns.append(word_index[w])
        X.append(ns)

    dataset = (X, labels) 
    print 'Saving...'
    cPickle.dump(dataset, open('data/reuters.pkl', 'w'))




def load_data(path="data/reuters.pkl", n_words=100000, maxlen=None, test_split=0.2, seed=113):
    path = get_file(path, origin="https://s3.amazonaws.com/text-datasets/reuters.pkl")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    X, labels = cPickle.load(f)
    f.close()

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels

    X = [[1 if w >= n_words else w for w in x] for x in X]
    X_train = X[:int(len(X)*(1-test_split))]
    y_train = labels[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = labels[int(len(X)*(1-test_split)):]

    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    make_reuters_dataset()
