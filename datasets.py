import pandas
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from transformers import BertTokenizer


def collate_fn(batch):
    x, y, z, r = zip(*batch)

    x = (torch.tensor(i) for i in x)
    y = (torch.tensor(i) for i in y)
    z = (torch.tensor(i) for i in z)

    x_pad = pad_sequence(x, batch_first=True)  # [b, N]
    y_pad = pad_sequence(y, batch_first=True)  # [b, N]
    z_pad = pad_sequence(z, batch_first=True)  # [b, N]

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
        pd = pandas.read_csv(self.path)
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

            # dict = {"room_type": room_type_embeddings,
            #         "travel_type": travel_type_embeddings,
            #         "text": text_embeddings,
            #         "rating": rating}
            return room_type_embeddings, travel_type_embeddings, text_embeddings, rating
        else:  # bert

            text_ids = self.tokenizer.encode(text)
            room_type_ids = self.tokenizer.encode(room_type)
            travel_type_ids = self.tokenizer.encode(travel_type)

            # dict = {"room_type": room_type_ids,
            #         "travel_type": travel_type_ids,
            #         "text": text_ids,
            #         "rating": rating}
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
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_dir)

    def _read_data(self, is_training):
        pd = pandas.read_excel(self.path, usecols=[0, 1, 2, 3])
        return pd

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        manufacturer = self.data.iloc[id, 0]
        category = self.data.iloc[id, 1]
        rating = self.data.iloc[id, 2]
        text = self.data.iloc[id, 3]

        # normalizing the score into (0, 1), assuming the highest score is 5
        rating = float(rating.split(" ")[0]) / 5

        if self.embedding_type == "glove":

            text_embeddings = self.GLOVE.get_vecs_by_tokens(self.tokenizer(text))
            manufacturer_embeddings = self.GLOVE.get_vecs_by_tokens(self.tokenizer(manufacturer))
            category_embeddings = self.GLOVE.get_vecs_by_tokens(self.tokenizer(category))

            # dict = {"manufacturer": manufacturer_embeddings,
            #         "category": category_embeddings,
            #         "text": text_embeddings,
            #         "rating": rating}
            return manufacturer_embeddings, category_embeddings, text_embeddings, rating

        else:

            text_ids = self.tokenizer.encode(text)
            manufacturer_ids = self.tokenizer.encode(manufacturer)
            category_ids = self.tokenizer.encode(category)

            # dict = {"manufacturer": manufacturer_ids,
            #         "category": category_ids,
            #         "text": text_ids,
            #         "rating": rating}
            return manufacturer_ids, category_ids, text_ids, rating


if __name__ == "__main__":
    from easydict import EasyDict

    config = {"dataset_path": "./data/hotel.csv",
              "embedding_type": "bert",
              "bert_model_dir": "../HuggingFaceH4/bert-base-chinese"
              }
    # config = {"dataset_path": "./data/amazon.xlsx",
    #           "embedding_type": "bert",
    #           "bert_model_dir": "../HuggingFaceH4/bert-base-uncased",
    #           "glove_dir": "/home/tye/code/HuggingFaceH4/"
    #           }
    config = EasyDict(config)

    ds = HotelDataset(config)
    # ds = AmazonDataset(config)

    train_dataloader = DataLoader(ds,  batch_size=2,  shuffle=False, collate_fn=collate_fn)

    for batch in train_dataloader:
        x, y, z, r = batch
        print(x.shape, y.shape, z.shape, r.shape)
