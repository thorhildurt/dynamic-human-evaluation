import os
import io
import json
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from utils import OrderedCounter
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class YELP(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)
        self.analyser = SentimentIntensityAnalyzer()

        ## qua già test train e validation sono spezzati quindi magari possiamo farlo noi che li spezziamo
        self.raw_data_path = os.path.join(data_dir, 'yelp_'+split+".csv")
        self.data_file = 'yelp_'+split+'.json'# TODO da vedere cosa fare
        self.vocab_file= 'yelp_vocab.json'

        if create_data:
            print("Creating new %s yelp data."%split.upper())
            self._create_data()
        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length'],
            'label': np.asarray(self.data[idx]['label'], dtype=np.float32)
        }

    @property
    def vocab_size(self):
        return len(self.w2i)
    @property
    def pad_idx(self):
        return self.w2i['<pad>']
    @property
    def sos_idx(self):
        return self.w2i['<sos>']
    @property
    def eos_idx(self):
        return self.w2i['<eos>']
    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i
    def get_i2w(self):
        return self.i2w


    def _load_vocab(self):

        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == "train":
            self._create_vocab()
        else:
            self._load_vocab()
        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        file = pd.read_csv(self.raw_data_path, nrows=1000)['review']
        file = file.dropna(axis=0)
        for i, line in enumerate(file):
            score = self.analyser.polarity_scores(line)
            score = [score['neg'], score['neu'], score['pos']]
            words = tokenizer.tokenize(line)

            input = ['<sos>'] + words
            input = input[:self.max_sequence_length]
            target = words[:self.max_sequence_length - 1]
            target = target + ['<eos>']

            assert len(input) == len(target), "%i, %i" % (len(input), len(target))

            length = len(input)
            input.extend(['<pad>'] * (self.max_sequence_length - length))
            target.extend(['<pad>'] * (self.max_sequence_length - length))

            input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
            target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

            id = len(data)
            data[id]['input'] = input
            data[id]['target'] = target
            data[id]['length'] = length
            data[id]['label'] = score

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _load_data(self, vocab=True):
        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."
        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)
        print("PATH: ",self.raw_data_path)
        file = pd.read_csv(self.raw_data_path)['review']
        print("Data size: ", file.shape)
        file = file.dropna(axis=0)
        for i, line in enumerate(file):
            if i == 27054:
                continue
            words = tokenizer.tokenize(line)
            w2c.update(words)

        for w, c in w2c.items():
            if c > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocabulary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()



