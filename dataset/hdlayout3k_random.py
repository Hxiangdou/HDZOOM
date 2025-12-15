import dataset.transformsHDLayout as T
from pathlib import Path
from PIL import Image
import os, json
from torch.utils.data.dataset import Dataset
import torch
import numpy as np

def random_bbox():
    list_points = np.random.rand(4) * 512
    x1, x2 = min(list_points[0], list_points[2]), max(list_points[0], list_points[2])
    y1, y2 = min(list_points[1], list_points[3]), max(list_points[1], list_points[3])
    return [[int(x1), int(y1)], [int(x2), int(y2)]]

def random_bezier():
    list_points = np.random.rand(16) * 512
    for t in list_points:
        t = int(t)
    return list_points.tolist()

class HDLayoutDataset(Dataset):
    def __init__(self, img_folder, json_file, transforms, num_queries):
        # dataload
        self.img_folder = img_folder
        self.json_folder = json_file
        self.block_bbox_path = os.path.join(self.json_folder, 'block')
        self.line_bbox_path = os.path.join(self.json_folder, 'line1')
        self.char_bezier_path = os.path.join(self.json_folder, 'line2')
        # transforms
        self._transforms = transforms
        # data
        self.data = []
        self.img_path = []
        for json_file in os.listdir(self.char_bezier_path):
            total_block_path = os.path.join(self.block_bbox_path, json_file)
            total_line1_path = os.path.join(self.line_bbox_path, json_file)
            total_line2_path = os.path.join(self.char_bezier_path, json_file)
            total_img_path = os.path.join(self.img_folder, json_file.replace('.json', '.jpg'))
            # load json
            try:
                with open(total_block_path, 'r') as file:
                    block_data = json.load(file)
                with open(total_line1_path, 'r') as file:
                    line1_data = json.load(file)
                with open(total_line2_path, 'r') as file:
                    line2_data = json.load(file)
            except:
                print(f'{json_file} json file error.')
                continue
            
            h, w = float(line2_data['imageHeight']), float(line2_data['imageWidth'])
            char_bezier, line_bbox, block_bbox, char_bezier_label, line_bbox_label, block_bbox_label = self.load_json(line2_data, line1_data, block_data, num_queries)
            if not (char_bezier != [] and line_bbox != [] and block_bbox != []):
                print(f'{json_file} json file error. char_bezier:{len(char_bezier)}, {len(line_bbox)}, {len(block_bbox)}')
                continue
            
            target = {
                'char_bezier': char_bezier,
                'line_bbox': line_bbox,
                'block_bbox': block_bbox,
                'char_bezier_labels': torch.tensor(char_bezier_label, dtype=torch.int64),
                'line_bbox_labels': torch.tensor(line_bbox_label, dtype=torch.int64),
                'block_bbox_labels': torch.tensor(block_bbox_label, dtype=torch.int64),
                'orig_size': [h, w],
            }
            # load image
            img = Image.open(total_img_path).convert('RGB')
            
            # transform
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            
            # samples = {
            #     'img': img,
            #     'block_bbox': target['block_bbox'],
            # }
            # samples_block_bbox = target['block_bbox']
            # _ = target.pop('block_bbox')
            
            
            # labels = {
            #         'char_bezier_labels': torch.ones((len(char_bezier),), dtype=torch.int64),
            #         'line_bbox_labels': torch.ones((len(line_bbox),), dtype=torch.int64),
            #         'block_bbox_labels': torch.ones((len(block_bbox),), dtype=torch.int64),
            #     }
            # target.update(labels)
            self.data.append((img, target))
            self.img_path.append(total_img_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]
        return img, target, self.img_path[idx]
    
    def load_json(self, line2_data, line1_data, block_data, num_queries):
        # num_queries = [num_queries[i] * num_queries[i-1] for i in range(1, len(num_queries))]
        # 存储结构
        char_bezier, line_bbox, block_bbox = [], [], []
        # 标记哪些line1、block已经被加载
        drawn_line1 = set()
        drawn_block = set()
        char_bezier_label, line_bbox_label, block_bbox_label = [], [], []
        # 加载对应的line2
        for line2_shape in line2_data['shapes']:
            line1_label = line2_shape.get('bbox_label')
            if line1_label is not None:
                if len(line2_shape['points']) < 16:
                    print('line2data num error.')
                    continue
                if line1_label not in drawn_line1:
                    # 加载对应的line1
                    for line1_shape in line1_data['shapes']:
                        if line1_shape['label'] == line1_label:
                            block_label = line1_shape.get('region_label')
                            if block_label is not None and len(line_bbox) < num_queries[-2]:
                                if len(line1_shape['points']) < 2:
                                    print('line1data num error')
                                    continue
                                if block_label not in drawn_block:
                                    # 加载对应的block
                                    if len(drawn_block) != 0:
                                        continue
                                    for block_shape in block_data['shapes']:
                                        if block_shape['label'] == block_label and len(block_bbox) < num_queries[-3]:
                                            # real data
                                            drawn_block.add(block_label)
                                            drawn_line1.add(line1_label)
                                            block_bbox.append(block_shape['points'][:2])
                                            block_bbox_label.append(1)
                                            line_bbox.append(line1_shape['points'][:2])
                                            line_bbox_label.append(1)
                                            char_bezier.append(line2_shape['points'][:16])
                                            char_bezier_label.append(1)
                                            
                                            # fake data
                                            block_bbox.append(random_bbox())
                                            block_bbox_label.append(0)
                                            line_bbox.append(random_bbox())
                                            line_bbox_label.append(0)
                                            char_bezier.append(random_bezier())
                                            char_bezier_label.append(0)
                                            
                                else:
                                    drawn_line1.add(line1_label)
                                    line_bbox.append(line1_shape['points'][:2])
                                    line_bbox_label.append(1)
                                    char_bezier.append(line2_shape['points'][:16])
                                    char_bezier_label.append(1)
                                    
                                    # fake data
                                    line_bbox.append(random_bbox())
                                    line_bbox_label.append(0)
                                    char_bezier.append(random_bezier())
                                    char_bezier_label.append(0)
                else:
                    char_bezier.append(line2_shape['points'][:16])
                    char_bezier_label.append(1)
                    
                    # fake data
                    char_bezier.append(random_bezier())
                    char_bezier_label.append(0)

        return char_bezier, line_bbox, block_bbox, char_bezier_label, line_bbox_label, block_bbox_label
    
def HDLTransforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    scales = [512]  # Since the image is 512x512, we can only resize it to this scale
    
    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomMove(),
            normalize,
        ])
    else:
        return normalize
    return normalize

def buildDataset(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "jsons" ),
        "val": (root / "val" / "images", root / "val" / "jsons" ),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = HDLayoutDataset(
        img_folder, 
        ann_file, 
        transforms=HDLTransforms(image_set), 
        num_queries=args.num_queries)
    return dataset