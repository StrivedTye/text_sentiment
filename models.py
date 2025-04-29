import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from fusion import RelationAttention, P2B_XCorr
from anfis.membership import make_gauss_mfs, make_anfis
import pytorch_utils as ptu


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
        self.fusion = P2B_XCorr(config.embedding_dim, 128, 256)

        layers = [nn.Linear(hidden_size, hidden_size),
                  nn.ReLU(),
                  nn.Linear(hidden_size, 1)
                  ]
        self.classifier = nn.Sequential(*layers)

    def forward(self, p_ids, x_ids, y_ids, h, z_ids):
        # product_ids, manufacturer_ids, category_ids, helpfulness, text_ids
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


class ComparisonModel(nn.Module):
    def __init__(self, args, which_model='cnn', which_embeding='glove', embedding_dim=300):
        super(ComparisonModel, self).__init__()

        self.which_model = which_model
        self.which_embeding = which_embeding
        self.Z_H = nn.Linear(embedding_dim+1, embedding_dim)

        if which_embeding == 'bert':
            model_config = BertConfig.from_pretrained(args.bert_model_dir)
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
            self.bert = BertModel.from_pretrained(args.bert_model_dir, config=model_config)

        if which_model == 'cnn':
            # CNN
            self.cnn = nn.Sequential(
                ptu.Conv1d(embedding_dim, 128, kernel_size=5, activation=nn.ReLU(), bn=True),
                nn.AdaptiveMaxPool1d(1),
                ptu.Conv1d(128, 1, kernel_size=1, activation=None, bn=False),
                nn.Sigmoid()
            )

        elif which_model == 'lstm':
            # LSTM
            self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=128, num_layers=2, batch_first=True)
            self.lstm_fc = nn.Sequential(
                ptu.FC(128, 1, activation=None, bn=False),
                nn.Sigmoid()
            )

        else:
            # fully connection
            self.fc = nn.Sequential(
                ptu.FC(128, 128, activation=nn.ReLU(), bn=True),
                ptu.FC(128, 1,  activation=None, bn=False),
                nn.Sigmoid()
            )

    def forward(self, p_embedding, x_embedding, y_embedding, h, z_embedding):
        # product_embeddings, manufacturer_embeddings, category_embeddings, helpfulness, text_embeddings

        if self.which_embeding == 'bert':
            p_ids, x_ids, y_ids, h, z_ids = p_embedding, x_embedding, y_embedding, h, z_embedding

            with torch.no_grad():
                # get room_type's features
                try:
                    p_embedding = self.bert(p_ids, token_type_ids=torch.zeros_like(p_ids),
                                            attention_mask=torch.ones_like(p_ids))[0]  # [(B, L, D), (B, D)]

                    x_embedding = self.bert(x_ids, token_type_ids=torch.zeros_like(x_ids),
                                            attention_mask=torch.ones_like(x_ids))[0]  # [(B, L, D), (B, D)]

                    # get review's features
                    z_embedding = self.bert(z_ids, token_type_ids=torch.zeros_like(z_ids),
                                            attention_mask=torch.ones_like(z_ids))[0]  # [(B, L, D), (B, D)]

                except RuntimeError:
                    for i in range(x_ids.shape[0]):
                        print(self.tokenizer.decode(x_ids[i, :]))

        if self.which_model == 'cnn':

            # CNN
            h_expanded = h.view(-1, 1, 1).repeat(1, z_embedding.size(1), 1)  # (B, L, 1)
            fusion_feat = torch.cat([z_embedding, h_expanded], dim=2)  # (B, L, D+1)
            fusion_feat = self.Z_H(fusion_feat)
            fusion_feat = torch.cat([p_embedding, x_embedding,  fusion_feat], dim=1)  # (B, 3L, D)
            output = self.cnn(fusion_feat.transpose(1, 2).contiguous())  # output: [B, 1, 1]
            output = output.squeeze(-1)

        elif self.which_model == 'lstm':

            # LSTM
            h_expanded = h.view(-1, 1, 1).repeat(1, z_embedding.size(1), 1)  # (B, L, 1)
            fusion_feat = torch.cat([z_embedding, h_expanded], dim=2)  # (B, L, D+1)
            fusion_feat = self.Z_H(fusion_feat)
            fusion_feat = torch.cat([p_embedding, x_embedding,  fusion_feat], dim=1)  # (B, 3L, D)
            output, _ = self.lstm(fusion_feat.transpose(1, 2).contiguous()) # [B, 3L, D]
            output = self.lstm_fc(output[:, -1, :])  # [B, 1]

        else:

            # FC
            h_expanded = h.view(-1, 1, 1).repeat(1, z_embedding.size(1), 1)  # (B, L, 1)
            fusion_feat = torch.cat([z_embedding, h_expanded], dim=2)  # (B, L, D+1)
            fusion_feat = self.Z_H(fusion_feat)
            fusion_feat = torch.cat([p_embedding, x_embedding,  fusion_feat], dim=1)  # (B, 3L, D)
            fusion_feat = F.avg_pool1d(fusion_feat.transpose(1, 2).contiguous(), fusion_feat.size(1))
            output = self.fc(fusion_feat.squeeze(-1))  # output: [B, 1]

        return output
