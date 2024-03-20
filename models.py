import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from fusion import RelationAttention, P2B_XCorr
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

        # self.fusion = RelationAttention(in_dim=2*config.hidden_size, hidden_dim=128)
        self.fusion = P2B_XCorr(config.hidden_size, 128, 256)

        layers = [nn.Linear(config.hidden_size, hidden_size),
                  nn.ReLU(),
                  nn.Linear(hidden_size, 1)
                  ]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_ids, y_ids, z_ids):
        with torch.no_grad():
            # get room_type's features
            try:
                x_embedding = self.bert(x_ids, token_type_ids=torch.zeros_like(x_ids),
                                        attention_mask=torch.ones_like(x_ids))[0]  # [(B, L, D), (B, D)]

                # get travel_type's features
                y_embedding = self.bert(y_ids, token_type_ids=torch.zeros_like(y_ids),
                                        attention_mask=torch.ones_like(y_ids))[0]  # [(B, L, D), (B, D)]

                # get review's features
                z_embedding = self.bert(z_ids, token_type_ids=torch.zeros_like(z_ids),
                                        attention_mask=torch.ones_like(z_ids))[0]  # [(B, L, D), (B, D)]
            except RuntimeError:
                for i in range(x_ids.shape[0]):
                    print(self.tokenizer.decode(x_ids[i, :]))

        # fusion of three types of features
        aspect_feat = torch.cat([x_embedding, y_embedding], dim=1)  # [B, L, D]
        # mask = torch.ones_like(aspect_feat).mean(2)
        # fused_feats = self.fusion(z_embedding, aspect_feat, mask)  # [B, D]

        fused_feats = self.fusion(z_embedding.transpose(1, 2).contiguous(),
                                  aspect_feat.transpose(1, 2).contiguous(),)
        fused_feats = F.avg_pool1d(fused_feats, aspect_feat.size(1)).squeeze(2)  # [B, D]

        # conduct anfis: 900d->rating
        # self.anfis = make_anfis(fused_feats, num_mfs=3, num_out=1)
        # outputs = self.anfis(fused_feats)

        outputs = self.classifier(fused_feats)

        return outputs