import pandas
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer


def collate_fn(batch):
    x, y, z, r = zip(*batch)
    bs = len(x)

    x = [torch.tensor(i) for i in x]
    y = [torch.tensor(i) for i in y]
    z = [torch.tensor(i) for i in z]

    # xyz_pad = pad_sequence(x+y+z, batch_first=True)
    # x_pad = xyz_pad[:bs, :]
    # y_pad = xyz_pad[bs:2*bs, :]
    # z_pad = xyz_pad[2*bs:, :]
    # r = torch.tensor(r)
    # return x_pad, y_pad, z_pad, r

    x_pad = pad_sequence(x, batch_first=True)  # [b, N1]
    y_pad = pad_sequence(y, batch_first=True)  # [b, N2]
    z_pad = pad_sequence(z, batch_first=True)  # [b, N3]
    r = torch.tensor(r)
    return x_pad, y_pad, z_pad, r



class HotelDataset(Dataset):
    def __init__(self, config, is_training=True):

        self.path = config.dataset_path
        self.data = self._read_data()
        self.embedding_type = config.embedding_type

        if self.embedding_type == "glove":
            self.tokenizer = get_tokenizer('basic_english')
            self.GLOVE = GloVe(name='840B', dim=300, cache=config.glove_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_dir)

    def _read_data(self, is_training=True):
        # pd = pandas.read_csv(self.path)
        pd = pandas.read_excel(self.path, usecols=[0, 1, 2, 3])
        return pd

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):

        room_type = self.data.iloc[id, 0]
        travel_type = self.data.iloc[id, 1]
        rating = self.data.iloc[id, 2]
        text = self.data.iloc[id, 3]

        # normalizing the score into (0, 1), assuming the highest score is 5
        rating = float(rating) / 5

        if self.embedding_type == "glove":

            text_embeddings = self.GLOVE.get_vecs_by_tokens(self.tokenizer(text))
            room_type_embeddings = self.GLOVE.get_vecs_by_tokens(self.tokenizer(room_type))
            travel_type_embeddings = self.GLOVE.get_vecs_by_tokens(self.tokenizer(travel_type))

            return room_type_embeddings, travel_type_embeddings, text_embeddings, rating
        else:  # bert
            try:
                text_ids = self.tokenizer.encode(text)
                room_type_ids = self.tokenizer.encode(room_type)
                travel_type_ids = self.tokenizer.encode(travel_type)
            except ValueError:
                return self.__getitem__(0)

            return room_type_ids, travel_type_ids, text_ids, rating


class AmazonDataset(Dataset):

    def __init__(self, config, is_training=True):
        self.path = config.dataset_path
        self.data = self._read_data(is_training)
        self.embedding_type = config.embedding_type

        if self.embedding_type == "glove":
            self.tokenizer = get_tokenizer('basic_english')
            self.GLOVE = GloVe(name='840B', dim=300, cache=config.glove_dir)
        else:
            bert_config = BertConfig.from_pretrained(config.bert_model_dir)
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_dir, model_max_length=512)
            self.bert = BertModel.from_pretrained(config.bert_model_dir, config=bert_config)

    def _read_data(self, is_training):
        pd = pandas.read_excel(self.path, usecols=[0, 1, 2, 3])
        # pd = pd[pd.iloc[:, 2].is]
        return pd

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        manufacturer = self.data.iloc[id, 0]
        category = self.data.iloc[id, 1]
        rating = self.data.iloc[id, 2]
        text = self.data.iloc[id, 3]

        # normalizing the score into (0, 1), assuming the highest score is 5
        rating = float(rating) / 5

        if self.embedding_type == "glove":

            text_embeddings = self.GLOVE.get_vecs_by_tokens(self.tokenizer(text))
            manufacturer_embeddings = self.GLOVE.get_vecs_by_tokens(self.tokenizer(manufacturer))
            category_embeddings = self.GLOVE.get_vecs_by_tokens(self.tokenizer(category))

            return manufacturer_embeddings, category_embeddings, text_embeddings, rating

        else:
            try:
                text_ids = self.tokenizer.encode(text, truncation=True)
                manufacturer_ids = self.tokenizer.encode(manufacturer, truncation=True)
                category_ids = self.tokenizer.encode(category, truncation=True)

                # text_encoded = self.tokenizer(text, return_tensors='pt')
                # text_embeddings = self.bert(**text_encoded) #[1, L, D] [1, D]
                # text_embeddings = text_embeddings[0][0].detach().numpy()
                # manufacturer_encoded = self.tokenizer(manufacturer, return_tensors='pt')
                # manufacturer_embeddings = self.bert(**manufacturer_encoded)
                # manufacturer_embeddings = manufacturer_embeddings[0][0].detach().numpy()
                # category_encoded = self.tokenizer(category, return_tensors='pt')
                # category_embeddings = self.bert(**category_encoded)
                # category_embeddings = category_embeddings[0][0].detach().numpy()

            except ValueError:
                return self.__getitem__(0)

            return manufacturer_ids, category_ids, text_ids, rating
            # return manufacturer_embeddings, category_embeddings, text_embeddings, rating


if __name__ == "__main__":
    from easydict import EasyDict

    # config = {"dataset_path": "./data/hotel_cleaned.xlsx",
    #           "embedding_type": "bert",
    #           "bert_model_dir": "../HuggingFaceH4/bert-base-chinese"
    #           }
    config = {"dataset_path": "./data/amazon_cleaned.xlsx",
              "embedding_type": "bert",
              "bert_model_dir": "../HuggingFaceH4/bert-base-uncased",
              "glove_dir": "/home/tye/code/HuggingFaceH4/"
              }
    config = EasyDict(config)

    # ds = HotelDataset(config)
    ds = AmazonDataset(config)

    train_dataloader = DataLoader(ds,  batch_size=64,  shuffle=False, collate_fn=collate_fn)

    for batch in train_dataloader:
        x, y, z, r = batch
        print(x.shape, y.shape, z.shape, r.shape)
