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
        p, x, y, h, z, r = batch
        p, x, y, h, z, r = p.cuda(), x.cuda(), y.cuda(), h.cuda(), z.cuda(), r.cuda()

        with torch.no_grad():
            hat_r = model(p, x, y, h, z)  # [B, 1]

        diff = torch.norm(hat_r.squeeze(1) - r)
        predictions = torch.where(diff < 0.1, 1, 0)
        score = torch.sum(predictions)
        accuracy += score.item()
    accuracy /= dataset_len

    print(f'Accuracy: {accuracy}')
