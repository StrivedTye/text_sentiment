import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import AdamW

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    return optimizer


def train(args, model, train_dataloader, test_dataloader):
    tb_writer = SummaryWriter()

    # loss
    citerion = torch.nn.SmoothL1Loss()

    # optimizer
    if args.embedding_type == 'bert':
        optimizer = get_bert_optimizer(args, model)
    else:
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    set_seed(args)
    global_step = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        loss_sum = 0
        dataset_len = len(train_dataloader.dataset)

        for step, batch in enumerate(train_dataloader):
            x, y, z, r = batch# x: room, y: travel, z: review, r: rating
            x, y, z, r = x.cuda(), y.cuda(), z.cuda(), r.cuda()
            batchsize = x.shape[0]

            hat_r = model(x, y, z)  # [B, 1]
            loss = citerion(hat_r, r.unsqueeze(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss * batchsize

            global_step += 1
            tb_writer.add_scalar('train_loss', loss, global_step)

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

        # test
        accuracy = 0
        dataset_len = len(test_dataloader.dataset)
        model.eval()
        for _, batch in enumerate(test_dataloader):
            x, y, z, r = batch# x: room, y: travel, z: review, r: rating
            x, y, z, r = x.cuda(), y.cuda(), z.cuda(), r.cuda()

            with torch.no_grad():
                hat_r = model(x, y, z)  # [B, 1]

            diff = torch.norm(hat_r.squeeze(1) - r)
            predictions = torch.where(diff < 0.1, 1, 0)
            score = torch.sum(predictions)
            accuracy += score.item()
        accuracy /= dataset_len

        print(f'Accuracy: {accuracy}')