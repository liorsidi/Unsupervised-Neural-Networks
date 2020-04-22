
import math
import pandas as pd
from sklearn.decomposition import PCA
from tldextract import tldextract
from torch.utils.data import DataLoader
from data import url_reverse_transform


def save_exploration(dataset, model,name,args):
    """
    extract embedding and pca on the data and save it to csv
    :param dataset: Dataset pytorch class
    :param model: the trained model
    :param name: namve value for thr file name
    :param args: the model arguments
    """
    #we wish to iterate on all the data therefore the shuffle is False
    dataloader_train = DataLoader(dataset, batch_size=32, shuffle=False)
    dfs = []
    for data in dataloader_train:
        batch, y = data
        output, emb = model(batch)
        output_ = url_reverse_transform(output.data.numpy(), args)
        batch_ = url_reverse_transform(y.data.numpy(), args)
        df = pd.DataFrame()
        df['url'] = batch_
        df['output'] = output_
        for i in range(8):
            df['emb_{}'.format(i)] = emb[:, i]
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    emb_cols = ['emb_{}'.format(i) for i in range(8)]
    pca = PCA(2)
    embs = df[emb_cols].values
    #normalize the data for pca
    embs_norm = (embs - embs.min()) / (embs.max() - embs.min())
    pcs = pca.fit_transform(embs_norm)
    pcs = pd.DataFrame(pcs,columns=['pc1','pc2'])
    df_full = pd.concat([df, pcs],axis = 1)
    df_full.to_csv("data/{}.csv".format(name))

def domain_features(x):
    """
    extract manual features of the data for manual exploration
    based on this function we will have touch and feel of the data and how to represent it for the model
    """
    features = dict()
    tldex = tldextract.extract(str(x).lower().rstrip('.'))
    features['domain'] = tldex.subdomain if tldex.subdomain != '' else tldex.domain
    features['suffix'] = tldex.domain + "." + tldex.suffix if tldex.subdomain != '' else tldex.suffix
    if features['domain'] == 'www':
        tldex = tldextract.extract(str(features['suffix']).lower().rstrip('.'))
        features['domain'] = tldex.subdomain if tldex.subdomain != '' else tldex.domain
        features['suffix'] = tldex.domain + "." + tldex.suffix if tldex.subdomain != '' else tldex.suffix

    features['domain_len'] = len(features['domain'])

    for c in features['domain']:
        if c in features:
            features[c] += 1
        else:
            features[c] = 1

    features['entropy'] = 0
    for c in set(features['domain']):
        features['entropy'] += features[c]/features['domain_len'] * math.log(features[c]/features['domain_len'])
    features['entropy'] = -1*features['entropy']
    return pd.Series(features)

if __name__ == "__main__":
    benign = pd.read_csv("data/top-1m.csv", names=['rank', 'url']).head(10000)
    phishing = pd.read_csv("data/phishTank.csv").head(5000)
    phishing['label'] = "phishing"
    benign['label'] = "benign"
    urls = pd.concat([phishing, benign])
    urls_f = urls['url'].apply(lambda x: domain_features(x))
    urls_f.to_csv("data/urls_features.csv")