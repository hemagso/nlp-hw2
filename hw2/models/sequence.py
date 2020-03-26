import torch.nn as nn
import torch.nn.functional as F
from ..constants import PAD_INDEX
from .encoders import Encoder
from .decoders import BahdanauDecoder

class EncoderDecoder(nn.Module):
    """A Encoder-Decoder architecture with attention.
    """

    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, n_layers=1, bidirectional=False,
                 dropout=0.):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(embed_size, hidden_size, n_layers, bidirectional, dropout=dropout)
        self.decoder = BahdanauDecoder(
            embed_size,
            2*hidden_size if bidirectional else hidden_size,  # The number of hidden units for the decoder depends
            n_layers,                                         # on whether the encoder was bidirectional or not.
            dropout=dropout
        )
        self.src_embed = nn.Embedding(src_vocab_size, embed_size)
        self.trg_embed = nn.Embedding(trg_vocab_size, embed_size)
        self.generator = Generator(2*hidden_size if bidirectional else hidden_size, trg_vocab_size)

    def forward(self, src_ids, trg_ids, src_lengths):
        src_mask = (src_ids != PAD_INDEX)
        encoder_hiddens, encoder_finals = self.encode(src_ids, src_lengths)
        return self.decode(
            encoder_hiddens,
            encoder_finals,
            src_mask,
            trg_ids[:, :-1]
        )

    def encode(self, src_ids, src_lengths):
        return self.encoder(self.src_embed(src_ids), src_lengths)

    def decode(self, encoder_hiddens, encoder_finals, src_mask, trg_ids):
        return self.decoder(
            self.trg_embed(trg_ids),
            encoder_hiddens,
            encoder_finals,
            src_mask
        )


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)