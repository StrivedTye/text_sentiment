import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from fusion import RelationAttention
from anfis.anfis import AnfisNet
from anfis.membership import make_gauss_mfs, make_anfis


class Pure_Bert(nn.Module):
    '''
    Bert for sequence classification.
    '''

    def __init__(self, args, hidden_size=256):
        super(Pure_Bert, self).__init__()

        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(args.bert_model_dir, config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.fusion = RelationAttention(in_dim=600, hidden_dim=64)

        layers = [nn.Linear(config.hidden_size, hidden_size),
                  nn.ReLU(),
                  nn.Linear(hidden_size, args.num_classes)
                  ]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_ids, y_ids, z_ids):

        # get room_type's features
        token_type_ids = torch.zero_like(x_ids)
        x_embedding = self.bert(x_ids, token_type_ids=token_type_ids)[0] # [(B, L, D), (B, D)]

        # get travel_type's features
        token_type_ids = torch.zero_like(y_ids)
        y_embedding = self.bert(y_ids, token_type_ids=token_type_ids)[0]  # [(B, L, D), (B, D)]

        # get review's features
        token_type_ids = torch.zero_like(z_ids)
        z_embedding = self.bert(z_ids, token_type_ids=token_type_ids)[0]  # [(B, L, D), (B, D)]

        # fusion of three types of features
        aspect_feat = torch.cat([x_embedding, y_embedding], dim=1)  # [B, 2L, D]
        mask = torch.ones_like(aspect_feat).mean(2)
        fused_feats = self.fusion(z_embedding, aspect_feat, mask)  # [B, D]

        # conduct anfis: 900d->rating
        self.anfis = make_anfis(fused_feats, num_mfs=3, num_out=1)

        outputs = self.anfis(fused_feats)
        return outputs