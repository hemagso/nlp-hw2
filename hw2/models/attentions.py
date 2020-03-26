import torch.nn as nn
import torch.nn.functional as F
import torch


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()

        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        # QUERY = (BATCH_SIZE, 1, HIDDEN_SIZE)
        # PROJ_KEY = (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        # SCORES = (BATCH_SIZE, SEQ_LEN)
        query = self.query_layer(query)
        scores = self.value_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        mask = mask[:, :scores.shape[-1]]
        scores.data.masked_fill(mask == 0, -float("inf"))

        alphas = F.softmax(scores, dim=-1)
        context = torch.bmm(alphas, value)

        return context, alphas
