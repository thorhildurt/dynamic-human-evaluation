import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict
class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def to_var(x, cuda2, volatile=False):
    if torch.cuda.is_available():
        x = x.to(cuda2)
    return Variable(x, volatile=volatile)


def idx2word(idx, i2w, pad_idx):

    sent_str = [str()]*len(idx)

    for i, sent in enumerate(idx):

        for word_id in sent:

            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "

        sent_str[i] = sent_str[i].strip()


    return sent_str


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)

    return interpolation.T


def decoding_ouput(out, idx2word):


    out = out.detach().cpu().numpy()
    sentence_list = []
    for sent_idx in range(out.shape[0]):
        sentence = ""
        predict = out[sent_idx, :, :]
        keys = np.argmax(predict, axis=1)
        for key in keys:
            sentence += idx2word[str(key)] + " "
        sentence_list.append(sentence)

    return sentence_list

def expierment_name(args, ts):

    exp_name = str()
    exp_name += "BS=%i_"%args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_"%args.embedding_size
    exp_name += "%s_"%args.rnn_type.upper()
    exp_name += "HS=%i_"%args.hidden_size
    exp_name += "L=%i_"%args.num_layers
    exp_name += "BI=%i_"%args.bidirectional
    exp_name += "LS=%i_"%args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_"%args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_"%args.x0
    exp_name += "TS=%s"%ts

    return exp_name
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    return score['neg'], score['neu'], score['pos'], score['compound']

def sentiment_labeler(data):
    labels = np.zeros(4)
    labels = np.reshape(labels,(1,4))
    for idx, s in enumerate(data):
        print(idx)

        l = np.asarray(sentiment_analyzer_scores(s))
        l = np.reshape(l, (1, 4))

        labels = np.vstack((labels, l))

    labels = np.delete(labels, 0, 0)
    return labels
'''
sentiment_analyzer_scores("The phone is super cool.")
import pandas as pd
data = pd.read_csv("data/yelp_train.csv")
data = data['review'].values
data = np.delete(data, 27054, 0)
labels = sentiment_labeler(data)

'''

