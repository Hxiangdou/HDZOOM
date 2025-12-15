import argparse
import random
import time
from pathlib import Path
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
# import utils.misc as utils
# from utils.plotUtils import ShowLabelInference
# from utils.plotUtils import json_save, Line12Block, Line22Line1

from PIL import Image
import dataset.transforms as T
from utils.process_output import process_outputs_best_one
from models.dino_hdlayout_v4 import DINO_HDLayout, PostProcess
from utils.showLabelSample import render_generate_hdlayout
from infer_creatidesign_hdzoom import hdzoom_inference, hdzoom_load_model
from models.config import Config as hdlayout_config
from input_config import Config as input_config

def get_args_parser():
    parser = argparse.ArgumentParser('Set DINO_HDLayout model', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=[1, 4, 1], type=list,
                        help="Number of query slots")
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--intermediate', action='store_true')
    parser.add_argument('--pre_norm', action='store_true')
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_points', default=5, type=float,
                        help="Points coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--overlap_loss_coef', default=1, type=float)
    parser.add_argument('--prob_loss_coef', default=1, type=float)
    parser.add_argument('--point_loss_coef', default=5, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--font_path', default="font/Arial_Unicode.ttf", type=str)
    parser.add_argument('--img_path', default="/data/LATEX-EnxText-new/val/images", type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='/data/model/HDLayout/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--resume', default='/data/fength/HDDiffuser-Text/outputs/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def load_HDLayout(args, device):
    cfg = hdlayout_config()
    model = DINO_HDLayout(cfg, args)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters // 1024 // 1024, 'MB')

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model_without_ddp.load_state_dict(checkpoint['model'])
    model.eval()
    
    postprocessor = PostProcess()
    return model, postprocessor

def main(args):
    config = input_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16
    
    output_dir = Path(args.output_dir)
    time_now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = Path(args.output_dir) / time_now
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = torch.device(args.device)

    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # load model
    HDLayout, postprocessor = load_HDLayout(args, device)
    
    HDZoom_pipe = hdzoom_load_model(
        model_path=config.model_path,
        ckpt_repo=config.ckpt_repo,
        weight_dtype=weight_dtype,
        resolution=config.resolution
    )
    
    # HDLayout Layout
    
    json_res = []
    transforms = T.ImageCompose([
                T.ImageToTensor(),
                T.ImageNormalize(mean=[0.489, 0.456, 0.416], std=[0.229, 0.224, 0.225])
            ])
    img = Image.open(config.my_condition_image_path).convert("RGB")
    img = transforms(img).to(device).unsqueeze(0)
    # mask = torch.ones((img.shape[2], img.shape[3]), dtype=torch.bool, device=device)
    # mask[: img.shape[2], :img.shape[3]] = False
    # mask = mask.unsqueeze(0)
    
    # samples = utils.NestedTensor(img, mask)
    outputs = HDLayout(img)

    orig_target_sizes = torch.stack([torch.tensor([512, 512]) for t in range(1)], dim=0).to(device)
    result = postprocessor(outputs, orig_target_sizes)
    json_res.append({
        "img_path": args.img_path,
        "results": result[0]
    })
    outputs = {k: v.cpu().detach().numpy() for k, v in outputs.items()}
    # render
    if not os.path.exists(config.temp_path):
        os.makedirs(config.temp_path)
    processed_output = process_outputs_best_one(img=config.my_condition_image_path, outputs=outputs, prompt=config.my_prompt)
    my_condition_img = os.path.join(config.temp_path, "condition_img.jpg")
    processed_output['my_condition_img'].save(my_condition_img)
    mask_path = os.path.join(config.temp_path, "bezier_mask.jpg")
    processed_output['mask_img'].save(mask_path)
    my_layout = processed_output['my_layout']
    
    # print(f"inference result save: {output_dir}")
    # json_path = os.path.join(output_dir, 'jsons')
    # img_path = os.path.join(output_dir, 'imgs')
    # os.makedirs(json_path, exist_ok=True)
    # line2Block, bezier2Line, showLabelInference = Line12Block(), Line22Line1(), ShowLabelInference()
    # block_path, line_path, bezier_path = json_save(json_res, json_path)
    # line2Block.process(line_path, block_path, line_path)
    # bezier2Line.process(bezier_path, line_path, bezier_path)
    
    # HDZOOM
    
    hdzoom_inference(
        model_path=config.model_path,
        ckpt_repo=config.ckpt_repo,
        resolution=config.resolution,
        seed=config.seed,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        true_cfg_scale=config.true_cfg_scale,
        my_condition_image_path=my_condition_img, # bezier mask * img
        mask_path=mask_path, # bezier mask
        my_layout=my_layout, # line2 bbox
        my_prompt=config.my_prompt,
        output_dir=output_dir,
        device=device,
        weight_dtype=weight_dtype,
        pipe=HDZoom_pipe
    )
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)