import torch
import torch.nn as nn
torch.manual_seed(0)

class autoencoder(nn.Module):
    """
    a simple fully connected Autoencoder
    """
    def __init__(self, number_of_characters, batch_size=8, emb_size=20, max_length=30, **kwargs):
        super(autoencoder, self).__init__()
        self.number_of_characters = number_of_characters
        self.emb_size = emb_size
        self.max_length = max_length
        self.batch_size = batch_size

        self.Embedding = nn.Embedding(self.number_of_characters, 20)
        self.encoder = nn.Sequential(
            # nn.Linear(args['max_length'] * args['number_of_characters'], 128),
            nn.Linear(self.max_length*20, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 16),
            nn.ReLU(True),
            nn.Linear(16, 8))
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, self.max_length*self.number_of_characters),
            )
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.Embedding(x)
        x = x.view(-1, self.max_length*20)
        x = self.encoder(x)
        emb = x.data.numpy()
        x = self.decoder(x)
        x = x.view(-1,self.max_length, self.number_of_characters)
        x = self.sm(x)
        return x, emb
