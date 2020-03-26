import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """
    Encoder class for the Sequence2Sequence model. This class processes the input sequence using an RNN (More
    specifically, a GRU) and output the hiddens_states to be used by the Decoder stage.
    """
    def __init__(self, embed_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.):
        """
        :param embed_size: Integer representing the size of the embeddings being used in the model (Will be used
            for the input size of the RNN)
        :param hidden_size: An int representing the size of the hidden state of the RNN
        :param num_layers: The number of layers of the RNN
        :param bidirectional: Boolean representing whether or not the RNN goes both ways.
        :param dropout: A float representing dropout rate during training.
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, inputs, lengths):
        """
        Implements the forward pass of this component.

        :param inputs: a 3D tensor of shape (batch_size, max_seq_length, embed_size) representing
            a batch of padded embedded word vectors of sentences.
        :param lengths: A 1D tensor of shape (batch_size,) representing the sequence lengths of inputs.
        :return:
            - outputs: A 3-D tensor of shape (batch_size, max_seq_length, hidden_size*n_directions) containing
                all hidden states of the last RNN layer for all time steps.
            - finals: A 3-D tensor of shape (num_layers, batch_size, hidden_size*n_directions) containing the
                hidden states for all layers on the last timestep.
        """

        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        outputs, finals = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        finals = finals.view(                   # Here I reshape the tensor to allow for separate indexing
            self.num_layers,                    # for num_layers and direction, which will be useful below
            2 if self.bidirectional else 1,     # to make the concatenation of the bidirectional layers
            -1,                                 # clearer
            self.hidden_size
        )
        if self.bidirectional:
            finals = torch.cat([finals[:, 0], finals[:, 1]], dim=2)  # Concatenate both directions
        else:
            finals = finals[:, 0]  # Eliminate the direction dimensions
        return outputs, finals
