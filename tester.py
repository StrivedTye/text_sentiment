import logging
import torch
logger = logging.getLogger(__name__)


def test(args, model, test_dataloader):

    # load parameter

    # test
    accuracy = 0
    dataset_len = len(test_dataloader.dataset)
    model.eval()
    for _, batch in enumerate(test_dataloader):
        x, y, z, r = batch  # x: room, y: travel, z: review, r: rating
        x, y, z, r = x.cuda(), y.cuda(), z.cuda(), r.cuda()

        with torch.no_grad():
            hat_r = model(x, y, z)  # [B, 1]

        diff = torch.norm(hat_r.squeeze(1) - r)
        predictions = torch.where(diff < 0.1, 1, 0)
        score = torch.sum(predictions)
        accuracy += score.item()
    accuracy /= dataset_len

    print(f'Accuracy: {accuracy}')
