class Config:
    device = 'cuda'
    img_size = 256
    batch_size = 8
    num_epochs = 10
    lr = 1e-4
    num_classes = 1  # number of region classes
    
    backbone_name = 'vit_small'
    backbone_pretrained = True
    backbone_pretrained_weight = '/data/fength/dino-out/checkpoint.pth'
    share_backbone = True

    region_bbox_dim = 4
    region_bbox_num = 4
    
    line_bbox_dim = 4
    line_bbox_num = 16
    
    char_bezier_dim = 16
    char_bezier_num = line_bbox_num
    
    w_region = 1.0
    w_line = 1.0
    w_bezier = 1.0
    w_chamfer = 5.0

    save_dir = './checkpoints'
