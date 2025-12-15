import argparse
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
import utils.misc as utils
from utils.plotUtils import ShowLabelInference
from utils.plotUtils import json_save, Line12Block, Line22Line1
from models.hdlayout import PostProcess
from PIL import Image
import dataset.transforms as T
from models.hdlayoutcat import HDLayout
from utils.showLabelSample import render_generate_hdlayout
import json

def get_args_parser():
    parser = argparse.ArgumentParser('Set HDLayout model', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
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
    # parser.add_argument('--dec_layers', default=2, type=int,
    #                     help="Number of encoding layers in the transformer")
    # parser.add_argument('--dec_layers_block', default=2, type=int,
    #                     help="Number of decoding layers in the transformer")
    # parser.add_argument('--dec_layers_line', default=2, type=int,
    #                     help="Number of decoding layers in the transformer")
    # parser.add_argument('--dec_layers_bezier', default=2, type=int,
    #                     help="Number of decoding layers in the transformer")
    
    # parser.add_argument('--block_memory_coef', default=1, type=float,
    #                     help="Number of decoding layers in the transformer")
    # parser.add_argument('--line_memory_coef', default=1.5, type=float,
    #                     help="Number of decoding layers in the transformer")
    # parser.add_argument('--bezier_memory_coef', default=3, type=float,
    #                     help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
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
    # parser.add_argument('--overlap_loss_coef', default=1, type=float)
    # parser.add_argument('--prob_loss_coef', default=1, type=float)
    # parser.add_argument('--point_loss_coef', default=5, type=float)
    # parser.add_argument('--bbox_loss_coef', default=5, type=float)
    # parser.add_argument('--giou_loss_coef', default=2, type=float)
    # parser.add_argument('--eos_coef', default=0.1, type=float,
    #                     help="Relative classification weight of the no-object class")

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
    parser.add_argument('--resume', default='/data/fength/hdlayoutv2/20250911_150952/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    return parser

def main(args):
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # for i in range(1, len(args.num_queries)):
    #     args.num_queries[i] *= args.num_queries[i-1]

    model = HDLayout(args)
    model.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # dataloader --------------------
    transforms = T.ImageCompose([
                 T.ImageToTensor(),
                 T.ImageNormalize(mean=[0.489, 0.456, 0.416], std=[0.229, 0.224, 0.225])
                ])

    # modelloader --------------------

    output_dir = Path(args.output_dir)
    time_now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = Path(args.output_dir) / time_now
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint = torch.load(args.resume, map_location='cpu')
    # checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if 'transformer' not in k and 'block_decoder' not in k}
    model_without_ddp.load_state_dict(checkpoint['model'])

    # inference --------------------
    model.eval()
    
    postprocessor = PostProcess()

    if len(args.img_path.split('.')) > 1:
        json_res = []
        img = Image.open(args.img_path).convert('RGB')
        img = transforms(img)
        img = img.to(device).unsqueeze(0)
        mask = torch.ones((img.shape[2], img.shape[3]), dtype=torch.bool, device=device)
        mask[: img.shape[2], :img.shape[3]] = False
        mask = mask.unsqueeze(0)
        samples = utils.NestedTensor(img, mask)
        outputs = model(samples)

        orig_target_sizes = torch.stack([torch.tensor([512, 512]) for t in range(1)], dim=0).to(device)
        result = postprocessor(outputs, orig_target_sizes)
        json_res.append({
            "img_path": args.img_path,
            "results": result[0]
        })

        # render --------------------
        print(f"inference result save: {output_dir}")
        json_path = os.path.join(output_dir, 'jsons')
        img_path = os.path.join(output_dir, 'imgs')
        os.makedirs(json_path, exist_ok=True)
        line12Block, line22Line1, showLabelInference = Line12Block(), Line22Line1(), ShowLabelInference()
        block_path, line1_path, line2_path = json_save(json_res, json_path)
        line12Block.process(line1_path, block_path, line1_path)
        line22Line1.process(line2_path, line1_path, line2_path)
    else:
        for img_file in os.listdir(args.img_path):
            json_res = []
            input_img_path = os.path.join(args.img_path, img_file)
            json_file = img_file.split('.')[0] + '.json'
            input_json_path = os.path.join(args.img_path.replace("images", "jsons/block"), json_file)
            img = Image.open(input_img_path).convert('RGB')
            img = transforms(img)
            img = img.to(device).unsqueeze(0)
            mask = torch.ones((img.shape[2], img.shape[3]), dtype=torch.bool, device=device)
            mask[: img.shape[2], :img.shape[3]] = False
            mask = mask.unsqueeze(0)
            samples = utils.NestedTensor(img, mask)
            
            if not os.path.exists(input_json_path):
                continue
            with open(input_json_path) as file:
                block_data = json.load(file)
            block_bbox = block_data['shapes'][0]['points'][:2]
            block_bbox = torch.tensor(block_bbox).to(device).view(1, -1)
            outputs = model({"imgs": samples,
                        "block_bboxes": block_bbox})

            orig_target_sizes = torch.stack([torch.tensor([512, 512]) for t in range(1)], dim=0).to(device)
            result = postprocessor(outputs, orig_target_sizes)
            result[0]['block_bbox'] = [block_bbox, torch.tensor([[1]]).to(device)]
            json_res.append({
                "img_path": input_img_path,
                "results": result[0]
            })

            # render --------------------
            print(f"inference result save: {output_dir}")
            json_path = os.path.join(output_dir, 'jsons')
            img_path = os.path.join(output_dir, 'imgs')
            if not os.path.exists(json_path):
                os.makedirs(json_path, exist_ok=True)
            line12Block, line22Line1, showLabelInference = Line12Block(), Line22Line1(), ShowLabelInference()
            block_path, line1_path, line2_path = json_save(json_res, json_path)
            # block_path = input_json_path
            line12Block.process(line1_path, block_path, line1_path, json_file)
            line22Line1.process(line2_path, line1_path, line2_path, json_file)
        torch.cuda.empty_cache()
    render_generate_hdlayout(res_files=output_dir, image_folder=args.img_path)
    print('Done!')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HDLayout inference', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)