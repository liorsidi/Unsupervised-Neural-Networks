
from CNNAutoencoder import CNNAutoencoder
from Autoencoder import autoencoder
from data import UrlDataset, get_args, url_reverse_transform
from exploration import save_exploration
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd

def train_alexa(model_class, sample=True, num_epochs=12, batch_size=10, max_length=32, learning_rate=1e-3):
    """
    a training process given a model class, config and data train the model
    :param model_class:
    :param sample: if True will train on small portion of the data, good for debug
    :param num_epochs:
    :param batch_size:
    :param max_length:
    :param learning_rate:
    :return:
    """
    if sample:
        X = pd.read_csv('data/top-1m.csv', names=['rank', 'url'])['url'].head(1000)
    else:
        X = pd.read_csv('data/top-1m.csv', names=['rank', 'url'])['url']
    args = get_args(X)
    args['ohe'] = False
    args['max_length'] = max_length
    dataset_train = UrlDataset(X.values, args)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # very cool trick to initialize a model class and his arguments
    model = model_class(**args)
    print(model)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader_train:
            batch, y = data
            # ===================forward=====================
            output, _ = model(batch)
            loss = criterion(output, y)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))
        # each epoch, visualize the embeddings on of the last batch
        output_ = url_reverse_transform(output.data.numpy(), args)
        batch_ = url_reverse_transform(y.data.numpy(), args)
        for b, o in zip(batch_, output_, ):
            print("{} -> {}".format(b, o))

    torch.save(model.state_dict(), 'model/{}.pth'.format(model_class.__name__))
    # apply the model on the data and save the embeddings
    save_exploration(dataset_train, model, model_class.__name__, args)

if __name__ == "__main__":
    train_alexa(autoencoder)
    train_alexa(CNNAutoencoder)
