import argparse
import rasterio
from rasterio.errors import RasterioIOError
from torch.utils.data.dataset import Dataset
import os
import random
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.yvit_seg_modeling import VisionTransformer as yViT_seg
from networks.yvit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.yvit_seg_modeling_L2HNet import L2HNet
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch.nn.functional as F
import utils 
import torch
import concurrent.futures

from th import filtered_labels


class TileInferenceDataset(Dataset):
    
    def __init__(self, fn, label, chip_size, stride, transform=None, windowed_sampling=False, verbose=False, label_transform=None):
        self.fn = fn
        self.label = label
        self.chip_size = chip_size
        
        self.transform = transform
        self.label_transform = label_transform
        self.windowed_sampling = windowed_sampling
        self.verbose = verbose
        
        with rasterio.open(self.fn) as f:
            height, width = f.height, f.width
            self.num_channels = f.count
            self.dtype = f.profile["dtype"]
            if not windowed_sampling: # if we aren't using windowed sampling, then go ahead and read in all of the data
                self.data = np.rollaxis(f.read(), 0, 3)
            
        self.chip_coordinates = [] # upper left coordinate (y,x), of each chip that this Dataset will return
        for y in list(range(0, height - self.chip_size, stride)) + [height - self.chip_size]:
            for x in list(range(0, width - self.chip_size, stride)) + [width - self.chip_size]:
                self.chip_coordinates.append((y,x))
        self.num_chips = len(self.chip_coordinates)

        if self.verbose:
            print("Constructed TileInferenceDataset -- we have %d by %d file with %d channels with a dtype of %s. We are sampling %d chips from it." % (
                height, width, self.num_channels, self.dtype, self.num_chips
            ))
            
    def __getitem__(self, idx):
        y, x = self.chip_coordinates[idx]
        label_fp = rasterio.open(self.label, "r")
        label_fp = label_fp.read().squeeze()
        if self.windowed_sampling:
            try:
                with rasterio.Env():
                    with rasterio.open(self.fn) as f:
                        img = np.rollaxis(f.read(window=rasterio.windows.Window(x, y, self.chip_size, self.chip_size)), 0, 3)
                        label = label_fp.read(window=rasterio.Window(x, y, self.chip_size, self.chip_size)).squeeze()
            except RasterioIOError as e: # NOTE(caleb): I put this here to catch weird errors that I was seeing occasionally when trying to read from COGS - I don't remember the details though
                print("Reading %d failed, returning 0's" % (idx))
                img = np.zeros((self.chip_size, self.chip_size, self.num_channels), dtype=np.uint8)
        else:
            img = self.data[y:y+self.chip_size, x:x+self.chip_size]
            label = label_fp[y:y+self.chip_size, x:x+self.chip_size]


        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label, np.array((y,x))
        
    def __len__(self):
        return self.num_chips

parser = argparse.ArgumentParser()
CHIP_SIZE = 224
PADDING = 112
assert PADDING % 2 == 0
HALF_PADDING = PADDING//2
CHIP_STRIDE = 112
parser.add_argument('--dataset', type=str, default='CP_ny', help='experiment_name')
parser.add_argument('--image_band', type=int, default=4, help='image-band')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--CNN_width', type=int, default=64, help='L2HNet_width_size, default is 64: light mode. Set to 128: normal mode')
# parser.add_argument('--save_path', type=str, default="./results/chesapeake", help='save_path')
parser.add_argument('--save_path', type=str, default='./results/CP/ny/2025-07-07_17-04-48/map')
parser.add_argument('--model_path', type=str, default="./results/CP/ny/2025-07-07_17-04-48/epoch_97.pth", help='model_path')
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--gpu', type=str, help='Select GPU number to train', default='0')
def inference(args, model, test_save_path=None):
    device = args.device
    model.eval()
    input_dataframe = pd.read_csv(args.list_dir)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
    for image_idx in range(len(image_fns)):
        image_fn = image_fns[image_idx]
        label_fn = label_fns[image_idx]

        print("(%d/%d) Processing %s" % (image_idx, len(image_fns), image_fn), end=" ... ")
        #-------------------
        # Load input and create dataloader
        #-------------------
        def image_transforms(img):
            img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
            img = np.rollaxis(img, 2, 0).astype(np.float32)
            img = torch.from_numpy(img)
            return img

        def label_transforms(labels):
            labels = utils.LABEL_CLASS_TO_IDX_MAP[labels]
            labels = torch.from_numpy(labels)
            return labels

        with rasterio.open(image_fn) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()
        
        dataset = TileInferenceDataset(image_fn, label_fn, chip_size=CHIP_SIZE, stride=CHIP_STRIDE, transform=image_transforms, verbose=False, label_transform=label_transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
        )

        #-------------------
        # Run model and organize output
        #-------------------

        output = np.zeros((args.num_classes, input_height, input_width), dtype=np.float32)
        kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
        kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        output1 = torch.zeros((input_height, input_width), dtype=torch.bool, device=device)

        output2 = np.zeros((input_height, input_width), dtype=np.float32)



        for i, (data, label, coords) in enumerate(dataloader):
            data = data.cuda(device)
            label = label.cuda(device)
            with torch.no_grad():
                t_output1, t_output2, ff = model(data, label)
                t_output = F.softmax((t_output1), dim=1)  # Created mask label
                t_output = t_output.argmax(axis=1)# map1 = F.softmax(t_output1, dim=1).cpu().numpy()
                # map2 = F.softmax(t_output2, dim=1).cpu().numpy()
                # t_output = F.softmax((0.6*t_output1 + 0.4*t_output2), dim=1).cpu().numpy()
                # t_output = F.softmax((0.7*t_output1 + 0.3*t_output2), dim=1).cpu().numpy()
                mask_output = filtered_labels(t_output1, ff, t_output, device)
                mask_output = mask_output > 0
                t_output = F.softmax(((0.5*t_output1+0.5*t_output2)), dim=1).cpu().numpy() # Fuse two branches outputs

            for j in range(t_output.shape[0]):
                y, x = coords[j]
                output[:, y:y+CHIP_SIZE, x:x+CHIP_SIZE] += t_output[j] * kernel
                counts[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += kernel

                output1[y:y + CHIP_SIZE, x:x + CHIP_SIZE] = mask_output[j]
                #
                # output2[:, y:y + CHIP_SIZE, x:x + CHIP_SIZE] += map2[j] * kernel
        
        output = output / counts
        output_hard = output.argmax(axis=0).astype(np.uint8)

        # output1 = output1 / counts
        # output_hard1 = output1.argmax(axis=0).astype(np.uint8)
        #
        # output2 = output2 / counts
        # output_hard2 = output2.argmax(axis=0).astype(np.uint8)

        #-------------------
        # Save output
        #-------------------
        output_profile = input_profile.copy()
        output_profile["driver"] = "GTiff"
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 1
        output_profile["nodata"] = 0

        output_fn = image_fn.split("/")[-1] 
        output_fn = output_fn.replace("naip-new", "predictions") # name the predictions
        output_fn = os.path.join(test_save_path, output_fn)

        # with rasterio.open(output_fn, "w", **output_profile) as f:
        #     f.write(output_hard, 1)
        #     f.write_colormap(1, utils.LABEL_IDX_COLORMAP)

        output_hard = utils.trans_label(output_hard, args.dataset)
        op = torch.where(output1, torch.tensor(output_hard, device=device), 0)
        op = op.cpu().numpy()
        # output_hard1 = utils.trans_label(output_hard1, args.dataset)
        # output_hard2 = utils.trans_label(output_hard2, args.dataset)

        def write_raster(output_fn, output_profile, output_data, colormap):
            # 创建 RGB 数据
            height, width = output_data.shape
            rgb_data = np.zeros((3, height, width), dtype=np.uint8)  # 创建一个三通道的数组

            # 填充 RGB 数据
            for value, color in colormap.items():
                rgb_data[0][output_data == value] = color[0]  # R 通道
                rgb_data[1][output_data == value] = color[1]  # G 通道
                rgb_data[2][output_data == value] = color[2]  # B 通道

            # 写入新的 RGB TIFF 文件
            output_profile.update({
                'count': 3,  # 三个波段
                'dtype': 'uint8'  # 数据类型
            })

            with rasterio.open(output_fn, 'w', **output_profile) as dst:
                dst.write(rgb_data)
            # with rasterio.open(output_fn, "w", **output_profile) as f:
            #     f.write(output_data, 1)
            #     f.write_colormap(1, colormap)

        output_fns = [
            output_fn.replace("map", "unify_map"),
            output_fn.replace("map", "t1"),
        ]

        # output_fn.replace("map", "t2")
        output_datas = [output_hard, op]  # , output_hard1, output_hard2
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(write_raster, fn, output_profile, data, utils.LABEL_IDX_UNIFY_COLORMAP) for fn, data in zip(output_fns, output_datas)]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    return "Testing Finished!"


if __name__ == "__main__":
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        microbatch=5,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    args.device = device
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'CP_de': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_de_test.csv', # The path of the *.csv file
            'num_classes': 17
        },
        'CP_md': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_md_test.csv', # The path of the *.csv file
            'num_classes': 17
        },
        'CP_ny': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_ny_test.csv', # The path of the *.csv file
            'num_classes': 17
        },
        'CP_pa': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_pa_test.csv', # The path of the *.csv file
            'num_classes': 17
        },
        'CP_va': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_va_test.csv', # The path of the *.csv file
            'num_classes': 17
        },
        'CP_wv': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_wv_test.csv', # The path of the *.csv file
            'num_classes': 17
        }
    }# Create a config to your own dataset here
    dataset_name = args.dataset
    image_band = args.image_band
    img_size = 224
    time_step = args.time_step
    snapshot = args.model_path
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    vit_patches_size=16
    config_vit = CONFIGS_ViT_seg["ViT-B_16"]
    config_vit.n_classes = args.num_classes
    config_vit.patches.size = (vit_patches_size, vit_patches_size)
    config_vit.patches.grid = (int(args.img_size/vit_patches_size), int(args.img_size/vit_patches_size))

    # net = yViT_seg(config_vit, backbone=L2HNet(width=args.CNN_width, image_band=image_band), img_size=args.img_size, num_classes=config_vit.n_classes).cuda(device)
    # weight = torch.load(snapshot, map_location=device)
    # net.load_state_dict(weight)

    rnd_gen = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn((args.batch_size, image_band, img_size, img_size), device=device, generator=rnd_gen)

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    net = ViT_seg(config_vit, backbone=model, img_size=img_size, num_classes=config_vit.n_classes, diffusion=diffusion,
                  time_step=time_step, noise=noise).cuda(device)
    weight = torch.load(snapshot, map_location=device)
    net.load_state_dict(weight, strict=False)

    test_save_path = args.save_path
    os.makedirs(test_save_path, exist_ok=True)
    os.makedirs(test_save_path.replace("map", "unify_map"), exist_ok=True)
    os.makedirs(test_save_path.replace("map", "t1"), exist_ok=True)
    os.makedirs(test_save_path.replace("map", "t2"), exist_ok=True)

    inference(args, net, test_save_path)


