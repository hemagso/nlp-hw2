import torch.nn as nn
import torch
from .attentions import BahdanauAttention


class BahdanauDecoder(nn.Module):
    """
    This class implements the Decoder for the Sequence2Sequence model. This implementation makes use of
    the attention mechanism proposed by (Bahdanau et al).
    """

    def __init__(self, embed_size, hidden_size, num_layers=1, dropout=0.):
        """
        :param embed_size: Integer representing the size of the embeddings being used in the model (Will be used
            for the input size of the RNN)
        :param hidden_size: An int representing the size of the hidden state of the RNN
        :param num_layers": The number of layers of the RNN
        :param dropout: A float representing dropout rate during training.
        """

        super(BahdanauDecoder, self).__init__()
        self.attention = BahdanauAttention(hidden_size)
        self.num_layers = num_layers

        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_layer = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.GRU(
            embed_size + hidden_size,   # The attention mechanism will concatenate the input with a weighted
            hidden_size,                # combination of the encoder states
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

    def forward(self, inputs, encoder_hiddens, encoder_finals, src_mask):
        """
        Implements the forward pass of the Decoder.

        :param inputs: a 3D tensor of shape (batch_size, max_seq_length, embed_size) representing
            a batch of word vectors of sentences.
        :param encoder_hiddens: a 3D tensor of shape (batch_size, max_seq_length, hidden_size*n_directions) containing
            the hidden states of the encoder for all time steps.
        :param encoder_finals: a 3D tensor of shape (num_layers, batch_size, hidden_size*n_directions) containing
            the hidden states of the encoder for all layers at the last time step.
        :param src_mask: a 3D tensor of shape (batch_size, max_seq_length) that identifies which positions in the source
            sequence are padding tokens.
        :return:
        """
        # The maximum number of steps to unroll the RNN.
        max_len = inputs.size(1)
        hidden = self.init_hidden(encoder_finals)
        outputs = []

        proj_key = self.attention.key_layer(encoder_hiddens)  # Calculated outside the attention class for
                                                              # efficiency
        for i in range(max_len):
            query = hidden[-1].unsqueeze(1)
            context, alphas = self.attention(query=query, proj_key=proj_key, value=encoder_hiddens, mask=src_mask)
            input_ = torch.cat((inputs[:, [i]], context), dim=2)
            output, hidden = self.rnn(input_, hidden)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)

        return hidden, outputs

    def init_hidden(self, encoder_finals):
        """Use encoder final hidden state to initialize decoder's first hidden
        state."""

        return encoder_finals

