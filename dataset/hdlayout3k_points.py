import dataset.transformsHDLayout as T
from pathlib import Path
from PIL import Image
import os, json
from torch.utils.data.dataset import Dataset
import torch

class HDLayoutDataset(Dataset):
    def __init__(self, img_folder, json_file, transforms, num_queries):
        # dataload
        self.img_folder = img_folder
        self.json_folder = json_file
        self.block_bbox_path = os.path.join(self.json_folder, 'block')
        self.line_bbox_path = os.path.join(self.json_folder, 'line1')
        self.char_point_path = os.path.join(self.json_folder, 'char')
        # transforms
        self._transforms = transforms
        # data
        self.data = []
        self.img_path = []
        for json_file in os.listdir(self.char_point_path):
            total_block_path = os.path.join(self.block_bbox_path, json_file)
            total_line_path = os.path.join(self.line_bbox_path, json_file)
            total_char_path = os.path.join(self.char_point_path, json_file)
            total_img_path = os.path.join(self.img_folder, json_file.replace('.json', '.jpg'))
            # load json
            try:
                with open(total_block_path, 'r') as file:
                    block_data = json.load(file)
                with open(total_line_path, 'r') as file:
                    line_data = json.load(file)
                with open(total_char_path, 'r') as file:
                    char_data = json.load(file)
            except:
                print(f'{json_file} json file error.')
                continue
            
            h, w = float(char_data['imageHeight']), float(char_data['imageWidth'])
            char_point, line_bbox, block_bbox = self.load_json(char_data, line_data, block_data, num_queries)
            if not (char_point != [] and line_bbox != [] and block_bbox != []):
                print(f'{json_file} json file error. char_point:{len(char_point)}, {len(line_bbox)}, {len(block_bbox)}')
                continue
            
            target = {
                'char_bezier': char_point,
                'line_bbox': line_bbox,
                'block_bbox': block_bbox,
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
            labels = {
                    'char_bezier_labels': torch.ones((len(char_point),), dtype=torch.int64),
                    'line_bbox_labels': torch.ones((len(line_bbox),), dtype=torch.int64),
                    'block_bbox_labels': torch.ones((len(block_bbox),), dtype=torch.int64),
                }
            target.update(labels)
            self.data.append((img, target))
            self.img_path.append(total_img_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]
        return img, target, self.img_path[idx]
    
    def load_json(self, char_data, line_data, block_data, num_queries):
        # num_queries = [num_queries[i] * num_queries[i-1] for i in range(1, len(num_queries))]
        # 存储结构
        char_point, line_bbox, block_bbox = [], [], []
        # 标记哪些line1、block已经被加载
        drawn_line = set()
        drawn_block = set()
        # 加载对应的line2
        for char_shape in char_data['shapes']:
            line_label = char_shape.get('bbox_label')
            if line_label is not None:
                if len(char_shape['points']) < 16:
                    print('line2data num error.')
                    continue
                if line_label not in drawn_line:
                    # 加载对应的line1
                    for line_shape in line_data['shapes']:
                        if line_shape['label'] == line_label:
                            block_label = line_shape.get('region_label')
                            if block_label is not None and len(line_bbox) < num_queries[-2]:
                                if len(line_shape['points']) < 2:
                                    print('line1data num error')
                                    continue
                                if block_label not in drawn_block:
                                    # 加载对应的block
                                    if len(drawn_block) != 0:
                                        continue
                                    for block_shape in block_data['shapes']:
                                        if block_shape['label'] == block_label and len(block_bbox) < num_queries[-3]:
                                            drawn_block.add(block_label)
                                            drawn_line.add(line_label)
                                            block_bbox.append(block_shape['points'][:2])
                                            line_bbox.append(line_shape['points'][:2])
                                            char_point.append(char_shape['points'][:16])
                                else:
                                    drawn_line.add(line_label)
                                    line_bbox.append(line_shape['points'][:2])
                                    char_point.append(char_shape['points'][:16])
                else:
                    char_point.append(char_shape['points'][:16])

        return char_point, line_bbox, block_bbox
    
def HDLTransforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    scales = [512]  # Since the image is 512x512, we can only resize it to this scale
    
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
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