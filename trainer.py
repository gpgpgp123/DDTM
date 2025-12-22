import logging
import math
import os
import random
import sys
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import utils
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioError
from torch.utils.data.dataset import IterableDataset
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from th import filtered_labels, compute_entropy_th, compute_separation_th


class BootstrapCrossEntropy(nn.Module):
    def __init__(self, beta=0.95, is_hard=0):
        super(BootstrapCrossEntropy, self).__init__()
        self.beta = beta
        self.is_hard = is_hard
        self.eps = 1e-7

    def forward(self, logits, labels):
        labels = F.one_hot(labels, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
        probs = F.softmax(logits, dim=1)
        probs = torch.clip(probs, self.eps, 1-self.eps)

        if self.is_hard:
            pred_label = F.one_hot(torch.argmax(probs, dim=1), num_classes=logits.shape[1]).permute(0, 3, 1, 2)
        else:
            pred_label = probs
        
        # Ignore class 0
        mask = (labels[:, 0] == 0).unsqueeze(1).expand_as(probs)
        probs = probs * mask
        pred_label = pred_label * mask
        labels = labels * mask

        loss = torch.sum((self.beta * labels + (1 - self.beta) * pred_label) * torch.log(probs), dim=1)
        loss = (- torch.sum(loss) / torch.sum(mask[:, 0]))
        return loss

class StreamingGeospatialDataset(IterableDataset):

    def __init__(self, imagery_fns, label_fns=None, groups=None, chip_size=256, num_chips_per_tile=200,
                 windowed_sampling=False, image_transform=None, label_transform=None, nodata_check=None, verbose=False):
        if label_fns is None:
            self.fns = imagery_fns
            self.use_labels = False
        else:
            self.fns = list(zip(imagery_fns, label_fns))

            self.use_labels = True

        self.groups = groups

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.windowed_sampling = windowed_sampling

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check

        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingGeospatialDataset")

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # We only want to shuffle the order we traverse the files if we are the first worker (else, every worker will shuffle the files...)
        if worker_id == 0:
            np.random.shuffle(self.fns)  # in place

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id + 1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):

            label_fn = None
            if self.use_labels:

                img_fn, label_fn = self.fns[idx]
            else:
                img_fn = self.fns[idx]

            if self.groups is not None:
                group = self.groups[idx]
            else:
                group = None

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn, label_fn, group)

    def stream_chips(self):
        for img_fn, label_fn, group in self.stream_tile_fns():
            num_skipped_chips = 0

            # Open file pointers
            img_fp = rasterio.open(img_fn, "r")
            label_fp = rasterio.open(label_fn, "r") if self.use_labels else None

            height, width = img_fp.shape
            if self.use_labels:  # garuntee that our label mask has the same dimensions as our imagery
                t_height, t_width = label_fp.shape
                assert height == t_height and width == t_width

            # If we aren't in windowed sampling mode then we should read the entire tile up front
            img_data = None
            label_data = None
            try:
                if not self.windowed_sampling:
                    img_data = np.rollaxis(img_fp.read(3), 0, 3)
                    if self.use_labels:
                        label_data = label_fp.read().squeeze()  # assume the label geotiff has a single channel
            except RasterioError as e:
                print("WARNING: Error reading in entire file, skipping to the next file")
                continue

            for i in range(self.num_chips_per_tile):
                # Select the top left pixel of our chip randomly
                x = np.random.randint(0, width - self.chip_size)
                y = np.random.randint(0, height - self.chip_size)

                # Read imagery / labels
                img = None
                labels = None
                if self.windowed_sampling:
                    try:
                        img = np.rollaxis(img_fp.read(window=Window(x, y, self.chip_size, self.chip_size)), 0, 3)
                        # print(img.shape)
                        if self.use_labels:
                            labels = label_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                    except RasterioError:
                        print("WARNING: Error reading chip from file, skipping to the next chip")
                        continue
                else:
                    img = img_data[y:y + self.chip_size, x:x + self.chip_size, :]
                    if self.use_labels:
                        labels = label_data[y:y + self.chip_size, x:x + self.chip_size]

                # Check for no data
                if self.nodata_check is not None:
                    if self.use_labels:
                        skip_chip = self.nodata_check(img, labels)
                    else:
                        skip_chip = self.nodata_check(img)

                    if skip_chip:  # The current chip has been identified as invalid by the `nodata_check(...)` method
                        num_skipped_chips += 1
                        continue

                # Transform the imagery
                if self.image_transform is not None:
                    if self.groups is None:
                        img = self.image_transform(img)
                    else:
                        img = self.image_transform(img, group)
                else:
                    img = torch.from_numpy(img).squeeze()

                # Transform the labels
                if self.use_labels:
                    if self.label_transform is not None:
                        if self.groups is None:

                            labels = self.label_transform(labels)
                        else:
                            print(label_fn)
                            labels = self.label_transform(labels, group)
                            print(labels)
                    else:
                        labels = torch.from_numpy(labels).squeeze()

                # Note, that img should be a torch "Double" type (i.e. a np.float32) and labels should be a torch "Long" type (i.e. np.int64)
                if self.use_labels:
                    yield img, labels
                else:
                    yield img
            # Close file pointers
            img_fp.close()
            if self.use_labels:
                label_fp.close()

            if num_skipped_chips > 0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())


class FocalLossMultiClass(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLossMultiClass, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, preds, labels):
        """
        preds: 模型的输出，未经过 Softmax，形状为 [N, C]
        labels: 标签，形状为 [N]，取值为类别的索引
        """
        eps = 1e-7
        # 计算 log-softmax
        preds_logsoft = F.log_softmax(preds, dim=1)
        # 取 softmax 概率值
        preds_softmax = torch.exp(preds_logsoft)
        # 选择真实类别对应的概率
        preds_softmax = preds_softmax.gather(1, labels.unsqueeze(1)).squeeze(1)
        preds_logsoft = preds_logsoft.gather(1, labels.unsqueeze(1)).squeeze(1)

        # 处理 alpha 参数
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                # 若 alpha 为列表或张量，则为类别权重
                alpha = self.alpha[labels]
            else:
                alpha = self.alpha
        else:
            alpha = 1.0

        # 计算 Focal Loss
        loss = -alpha * torch.pow(1 - preds_softmax, self.gamma) * preds_logsoft

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

def image_transforms(img):
    img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img


def label_transforms(labels):
    labels = utils.LABEL_CLASS_TO_IDX_MAP[labels]
    labels = torch.from_numpy(labels)
    return labels


def nodata_check(img, labels):
    return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)

def trainer_dataset(args, model, snapshot_path):
    device = args.device
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size

    # -------------------
    # Load input data
    # -------------------

    input_dataframe = pd.read_csv(args.list_dir)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
    NUM_CHIPS_PER_TILE = 50  # How many chips will be sampled from one large-scale tile
    CHIP_SIZE = 224  # Size of each sampled chip
    db_train = StreamingGeospatialDataset(
        imagery_fns=image_fns, label_fns=label_fns, groups=None, chip_size=CHIP_SIZE,
        num_chips_per_tile=NUM_CHIPS_PER_TILE, windowed_sampling=True, verbose=False,
        image_transform=image_transforms, label_transform=label_transforms, nodata_check=nodata_check
    )

    print("The length of train set is: {}".format(len(image_fns) * NUM_CHIPS_PER_TILE))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()
    ce_loss = CrossEntropyLoss(ignore_index=0)
    focal_loss = FocalLossMultiClass(0.5, 2)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    num_training_batches_per_epoch = int(len(image_fns) * NUM_CHIPS_PER_TILE / batch_size)
    max_iterations = args.max_epochs * len(image_fns) * NUM_CHIPS_PER_TILE
    logging.info(
        "{} iterations per epoch. {} max iterations ".format(len(image_fns) * NUM_CHIPS_PER_TILE, max_iterations))
    iterator = range(max_epoch)
    for epoch_num in iterator:
        loss1 = []
        loss2 = []
        nan_num = 0
        start_th = [0.95] * 16
        lamda = 0.3
        quantile = 0.1 + 0.2 * torch.exp(torch.tensor(-0.1 * epoch_num / 100, device=device))
        for i_batch, (image_batch, label_batch) in tqdm(enumerate(trainloader), total=num_training_batches_per_epoch):

            image_batch, label_batch = image_batch.cuda(device), label_batch.cuda(device)
            outputs1, outputs2, fuse_feature = model(image_batch, label_batch)
            t_output = F.softmax((outputs1), dim=1)  # Created mask label
            t_output = t_output.argmax(axis=1)

            mask_output = filtered_labels(outputs1, fuse_feature, t_output, device)

            loss_ce1 = ce_loss(outputs1, label_batch.long())
            loss_ce2 = ce_loss(outputs2, label_batch.long()) # mask_output[:].long()


            loss = 0.5 * loss_ce1 + 0.5 * loss_ce2
            # loss = loss_ce1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            loss1.append(loss_ce1.item())
            loss2.append(loss_ce2.item())
            iter_num = iter_num + 1
        avg_loss1 = np.nanmean(loss1)
        avg_loss2 = np.nanmean(loss2)
        logging.info('Epoch : %d, CE-branch1 : %f, MCE-branch2: %f, loss: %f, nan_num: %f' % (
        epoch_num, avg_loss1, avg_loss2, avg_loss1 * 0.5 + avg_loss2 * 0.5, nan_num)) # avg_loss1 * 0.5 + avg_loss2 * 0.5
        save_interval = 20
        # if epoch_num % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 5:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # iterator.close()
            if epoch_num == max_epoch - 1:
                break

    writer.close()
    return "Training Finished!"