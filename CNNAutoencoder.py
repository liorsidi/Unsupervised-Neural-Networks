import torch
import torch.nn as nn

torch.manual_seed(0)

class TimeDistributed(nn.Module):
    """
    Covolutional Autoencoder
    """
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class CNNAutoencoder(nn.Module):
    def __init__(self,number_of_characters,batch_size = 8,  emb_size = 20,max_length =30, **kwargs):
        super(CNNAutoencoder, self).__init__()
        self.number_of_characters = number_of_characters
        self.emb_size = emb_size
        self.max_length = max_length
        self.batch_size = batch_size

        # Character embedding of the class
        self.Embedding = nn.Embedding(self.number_of_characters, self.emb_size)

        # CNN encoders and decoders
        self.encoder = nn.Sequential(
            nn.Conv1d(self.max_length, 16, 3, stride=2, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),  # b, 16, 5, 5
            nn.Conv1d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 6, stride=2),  # b, 16, 6
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 32, 7, stride=3, padding=1),  # b, 32, 20
            nn.ReLU(True),
        )
        self.logits = nn.Linear(32*20, self.max_length * self.number_of_characters)

        # suppose to apply softmax for each time step separately, read more at https://keras.io/layers/wrappers/
        # TODO stil require testing
        self.sm = TimeDistributed(nn.Softmax())

    def forward(self, x):
        x = self.Embedding(x)
        x = self.encoder(x)
        emb = x.view(-1,8).data.numpy()
        x = self.decoder(x)
        x = x.view(-1 , self.max_length*20)
        x = self.logits(x)
        x = x.view(-1, self.max_length, self.number_of_characters)
        x = self.sm(x)
        return x, emb
