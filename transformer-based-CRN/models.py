import torch
import torch.nn as nn
import numpy as np
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, nums_treatment=4, nums_covariates=2, nums_outcome=1, nums_sequence_length=59, embedding_dim=8):
        super(TransformerModel, self).__init__()
        # model params
        self.input_feature_num = nums_treatment + nums_outcome + nums_covariates
        self.sequence_length = nums_sequence_length
        self.embedding_dim = embedding_dim

        # model archis
        # representation generation
        self.input_projection = nn.Linear(self.input_feature_num, self.embedding_dim)
        TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8,batch_first=True)
        self.TransformerEncoder = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=6)
        self.Encoder_mask = nn.Transformer.generate_square_subsequent_mask(self.sequence_length) # for time-series

        # y prediction
        self.y_predictor = nn.Linear(self.embedding_dim + nums_covariates + nums_treatment, nums_outcome)
        
        # x prediction
        self.x_predictor = nn.Linear(self.embedding_dim + nums_covariates, nums_treatment)

    def forward(self, history_input, current_covariates, current_treatments):
        
        history_input = self.input_projection(history_input)

        # history_input = history_input.transpose(0,1)
        self.Encoder_mask = self.Encoder_mask.to(history_input.device)
        encoder_representation = self.TransformerEncoder(history_input, mask = self.Encoder_mask) # previous historyrepresentation for [0, ..., T-1]

        # encoder_representation = history_input
        y_input = torch.cat((encoder_representation, current_covariates, current_treatments), axis=-1)
        x_input = torch.cat((encoder_representation, current_covariates), axis = -1)
        y_prediction = self.y_predictor(y_input)
        x_prediction = F.softmax(self.x_predictor(x_input),dim = -1) # logits to probability
        return y_prediction, x_prediction

if __name__ == '__main__':
    input_data = torch.zeros((10, 59, 7)).float()
    model = TransformerModel()
    output = model(input_data)
    print(output.shape)