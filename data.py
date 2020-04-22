from tldextract import tldextract
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


def url2Domain(x):
    """
    apply simple utl transformation to extract the domain name
    """
    x = str(x).lower().rstrip('.')
    tldex = tldextract.extract(x)
    x = tldex.subdomain if tldex.subdomain != '' else tldex.domain
    if x == 'www':
        suffix = tldex.domain + "." + tldex.suffix if tldex.subdomain != '' else tldex.suffix
        tldex = tldextract.extract(suffix)
        x = tldex.subdomain if tldex.subdomain != '' else tldex.domain
    return x


def url_reverse_transform(X, args):
    """
    reconstruct the url to string
    """
    valid_chars_rev = {v: k for k, v in args['vocabulary'].items()}
    valid_chars_rev[0] = ""
    X = [[np.argmax(c) for c in pred] for pred in X]
    urls = [''.join([valid_chars_rev[c] for c in x]) for x in X]
    return urls


def get_args(X):
    """
    compute the dataset arguments that relevant for data processing and model training
    :return: dictionary of argumants
    """
    X = X.apply(lambda u: url2Domain(u))

    # vocabulary of all the characters
    vocabulary = {x: idx + 1 for idx, x in enumerate(set(''.join(X)))}
    vocabulary[""] = 0
    vocabulary["<"] = len(vocabulary)
    vocabulary[">"] = len(vocabulary)
    number_of_characters = len(vocabulary)

    X = [[vocabulary[y] for y in x] for x in X]
    max_length = int(np.max([len(x) for x in X]))
    args = dict(vocabulary=vocabulary, number_of_characters=number_of_characters, max_length=max_length)
    return args


class UrlDataset(Dataset):
    """URL dataset"""
    def __init__(self, urls, args, transform=None, labels=None):
        self.vocabulary = args['vocabulary']
        self.max_length = args['max_length']
        self.number_of_characters = args['number_of_characters']
        self.input_ohe = args['ohe']
        self.urls = urls
        self.y = labels
        self.transform = transform
        self.identity_mat = np.identity(self.number_of_characters)

    def __len__(self):

        return self.urls.shape[0]

    def __getitem__(self, idx):

        x = self.urls[idx]

        x_domain = url2Domain(x)
        # add start and end characters to improve convergence
        x_domain = ">" + x_domain + "<"

        # represent the data as one hot encoded
        x_ohe = np.array(
            [self.identity_mat[self.vocabulary[i]] for i in list(x_domain)[::-1] if i in self.vocabulary.keys()],
            dtype=np.float32)
        dtype = np.float32
        if len(x_ohe) > self.max_length:
            x_ohe = x_ohe[:self.max_length]
        elif 0 < len(x_ohe) < self.max_length:
            x_ohe = np.concatenate(
                (x_ohe, np.zeros((self.max_length - len(x_ohe), self.number_of_characters), dtype=dtype)))
        elif len(x_ohe) == 0:
            x_ohe = np.zeros(
                (self.max_length, self.number_of_characters), dtype=dtype)

        # represent the data as list of integers
        x_ = np.array(
            [self.vocabulary[i] for i in list(x_domain)[::-1] if i in self.vocabulary.keys()],
            dtype=np.long)
        dtype = np.long
        if len(x_) > self.max_length:
            x_ = x_[:self.max_length]
        elif 0 < len(x_) < self.max_length:
            x_ = np.concatenate(
                (x_, np.zeros((self.max_length - len(x_)), dtype=dtype)))
        elif len(x_) == 0:
            x_ = np.zeros(
                (self.max_length), dtype=dtype)

        if self.y is None:
            y = x_ohe
        if self.input_ohe:
            x_ = x_ohe

        # the output of the network is ohe
        return x_, y

#just for testing
if __name__ == "__main__":
    X = pd.read_csv('data/top-1m.csv', names=['rank', 'url'])['url'].sample(1000)
    args = get_args(X)
    args['ohe'] = False
    url_dataset = UrlDataset(X.values, args)
    x, _ = url_dataset[53]
    print(x)
    print(url_reverse_transform([x], args))
