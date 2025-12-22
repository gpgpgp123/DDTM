import argparse
import random
from datetime import datetime

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling_stable import VisionTransformer as ViT_seg_stable
from networks.yvit_seg_modeling import VisionTransformer as yViT_seg
from networks.yvit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.yvit_seg_modeling_L2HNet import L2HNet
from trainer import trainer_dataset
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import os

def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # if device == torch.device("cuda"):
    #     model.cuda()
    # elif device == torch.device("cpu"):
    #     model.cpu()
    #     model.cond_stage_model.device = "cpu"
    # else:
    #     raise ValueError(f"Incorrect device name. Received: {device}")
    model.train()
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CP_wv', help='experiment_name')
parser.add_argument('--image_band', type=int, default=4, help='image-band')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--CNN_width', type=int, default=64, help='L2HNet_width_size, default is 64: light mode. Set to 128: normal mode')
parser.add_argument('--savepath', type=str, default='./results/CP/wv/')
parser.add_argument('--diffusion_path', type=str, default='./checkpoints/CP_diffusion/all/openai-2025-06-25-22-38-47-113680/model070000.pt')
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--gpu', type=str, help='Select GPU number to train', default='2')

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
    time_step = args.time_step
    vit_patches_size=16
    img_size=224
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    image_band = args.image_band
    dataset_config = {
        'CP_de': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_de_train.csv', # The path of the *.csv file
            'num_classes': 17
        },
        'CP_md': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_md_train.csv', # The path of the *.csv file
            'num_classes': 17
        },
        'CP_ny': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_ny_train.csv', # The path of the *.csv file
            'num_classes': 17
        },
        'CP_pa': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_pa_train.csv', # The path of the *.csv file
            'num_classes': 17
        },
        'CP_va': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_va_train.csv', # The path of the *.csv file
            'num_classes': 17
        },
        'CP_wv': {  # default dataset as a example
            'list_dir': './dataset/CSV_list/CP_wv_train.csv', # The path of the *.csv file
            'num_classes': 17
        }
    }# Create a config to your own dataset here
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_path = args.savepath + formatted_time
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_vit = CONFIGS_ViT_seg["ViT-B_16"]
    config_vit.n_classes = args.num_classes
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    # base
    # net = yViT_seg(config_vit, backbone=L2HNet(width=args.CNN_width),img_size=img_size, num_classes=config_vit.n_classes).cuda(device)

    # diffusion
    rnd_gen = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn((args.batch_size, image_band, img_size, img_size), device=device, generator=rnd_gen)

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # load diffusion model weight
    weight = torch.load(args.diffusion_path, map_location=device)
    model.load_state_dict(weight)

    net = ViT_seg(config_vit, backbone=model,img_size=img_size, num_classes=config_vit.n_classes, diffusion=diffusion, time_step=time_step, noise=noise).cuda(device)
    # freeze backbone
    for param in net.transformer.embeddings.hybrid_model.parameters():
        param.requires_grad = False

    net.load_from(weights=np.load(config_vit.pretrained_path))
    trainer_dataset(args, net, snapshot_path)