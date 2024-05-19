# coding=utf-8
import argparse
import logging
import os
import random
import numpy as np
import pandas as pd

import torch
from transformers import (BertConfig,
                          BertForTokenClassification,
                          BertTokenizer)
from torch.utils.data import DataLoader

from models import Pure_Bert
from trainer import train
from datasets import HotelDataset, AmazonDataset, collate_fn
from tester import test

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_name', type=str, default='hotel', help='Choose dataset.')
    parser.add_argument('--dataset_path', type=str, default='./data/hotel_cleaned.xlsx', help='dataset path.')

    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')

    parser.add_argument('--cuda_id', type=str, default='0', help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2024, help='random seed for initialization')

    parser.add_argument('--test', action='store_true', default=False, help='test stage')
    parser.add_argument('--checkpoint', type=str, default=None, help='pre-trained network')

    # Model parameters
    parser.add_argument('--glove_dir', type=str, default='/home/tye/code/HuggingFaceH4/',
                        help='Directory storing glove embeddings')
    parser.add_argument('--bert_model_dir', type=str, default='../HuggingFaceH4/bert-base-uncased',
                        help='Path to pre-trained Bert model.')

    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')

    parser.add_argument('--embedding_type', type=str, default='bert', choices=['glove', 'bert'])

    parser.add_argument('--hidden_size', type=int, default=768,
                        help='Hidden size of bert, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')

    # Training parameters
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")

    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations

    '''
    logger.info(vars(args))


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Parse args
    args = parse_args()
    check_args(args)

    # Setup CUDA, GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    # Load datasets
    if args.dataset_name == 'hotel':
        train_dataloader = DataLoader(HotelDataset(args, True), batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(HotelDataset(args, False), batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn)
    else: #'amazon'
        train_dataloader = DataLoader(AmazonDataset(args, True), batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(AmazonDataset(args, False), batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn)

    # Build Model
    model = Pure_Bert(args)
    model.to(args.device)

    #load previous network weights
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Using pretrain model.')
    else:
        logger.info('No existing model, starting training from scratch.')

    if args.train:
        train(args, model, train_dataloader, test_dataloader)
    else:
        test(args, model, test_dataloader)


if __name__ == "__main__":
    main()