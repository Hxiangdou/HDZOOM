"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path, PurePath
import os
import json
from boxOps import box_cxcywh_to_xyxy

def json_save(json_res, json_path):
    ''' save all level sperated results as json file'''
    json_paths = []
    res = {
        'block_bbox': [],
        'line_bbox': [],
        'char_bezier': [],
    }
    for img_item in json_res:
        for k, v in img_item['results'].items():
            json_v = {
                    'shapes': [],
                    "imagePath": None,
                    "imageData": None,
                    "imageHeight": 512,
                    "imageWidth": 512
                }
            for i, (point, score) in enumerate(zip(v[0], v[1])):
                json_v['shapes'].append({
                    'points': point.cpu().numpy().tolist(),
                    'score': round(score.cpu().item(), 2),
                    'label': i,
                })
            json_v['imagePath'] = img_item['img_path'].split('/')[-1]
            res[k].append(json_v)
    for k, v in res.items():
        json_path_sub = os.path.join(json_path, k)
        os.makedirs(json_path_sub, exist_ok=True)
        for json_item in v:
            json_item_path = os.path.join(json_path_sub, json_item['imagePath'].split('.')[0] + '.json')
            with open(json_item_path, 'w') as f:
                json.dump(json_item, f)
        json_paths.append(json_path_sub)
    return json_paths

class Line12Block():
    def __init__(self):
        pass

    def calculate_intersection_area(self, bbox1, bbox2):
        """计算两个bbox的交集面积"""
        x1_min, y1_min = bbox1[0]
        x1_max, y1_max = bbox1[1]
        
        x2_min, y2_min = bbox2[0]
        x2_max, y2_max = bbox2[1]
        
        intersect_min_x = max(x1_min, x2_min)
        intersect_min_y = max(y1_min, y2_min)
        intersect_max_x = min(x1_max, x2_max)
        intersect_max_y = min(y1_max, y2_max)
        
        intersect_width = max(0, intersect_max_x - intersect_min_x)
        intersect_height = max(0, intersect_max_y - intersect_min_y)
        
        return intersect_width * intersect_height

    def calculate_area(self, bbox):
        """计算bbox的面积"""
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        return (x_max - x_min) * (y_max - y_min)

    def calculate_intersection_area_other(self, bbox1, bbox2):
        """计算两个bbox的交集面积"""
        x1_min, y1_min = bbox1[0], bbox1[1]
        x1_max, y1_max = bbox1[2], bbox1[3]
        
        x2_min, y2_min = bbox2[0], bbox2[1]
        x2_max, y2_max = bbox2[2], bbox2[3]
        
        intersect_min_x = max(x1_min, x2_min)
        intersect_min_y = max(y1_min, y2_min)
        intersect_max_x = min(x1_max, x2_max)
        intersect_max_y = min(y1_max, y2_max)
        
        intersect_width = max(0, intersect_max_x - intersect_min_x)
        intersect_height = max(0, intersect_max_y - intersect_min_y)
        
        return intersect_width * intersect_height

    def calculate_area_other(self, bbox):
        """计算bbox的面积"""
        x_min, y_min = bbox[0], bbox[1]
        x_max, y_max = bbox[2], bbox[3]
        return (x_max - x_min) * (y_max - y_min)

    def process(self, line_bbox_folder, region_bbox_folder, output_folder, filename):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历第一个文件夹
        # for filename in os.listdir(line_bbox_folder):
        if filename.endswith('.json'):
            line_bbox_path = os.path.join(line_bbox_folder, filename)
            region_bbox_path = os.path.join(region_bbox_folder, filename)
            if not os.path.exists(region_bbox_path):
                print(f"Failed Processed {filename}")
                # continue
                return
            # 读取行字符包围盒JSON文件
            with open(line_bbox_path, 'r') as file:
                line_data = json.load(file)
            
            # 读取区域字符包围盒JSON文件
            with open(region_bbox_path, 'r') as file:
                region_data = json.load(file)
            
            # 遍历每个行字符包围盒
            for line_shape in line_data['shapes']:
                line_bbox = line_shape['points']
                if len(line_bbox) == 2:
                    line_area = self.calculate_area(line_bbox)
                elif len(line_bbox) == 4:
                    line_area = self.calculate_area_other(line_bbox)
                # 遍历每个区域字符包围盒
                for region_shape in region_data['shapes']:
                    region_bbox = region_shape['points']
                    
                    # 计算交集面积
                    if len(line_bbox) == 2:
                        intersection_area = self.calculate_intersection_area(line_bbox, region_bbox)
                    elif len(line_bbox) == 4:
                        intersection_area = self.calculate_intersection_area_other(line_bbox, region_bbox)
                    
                    # 判断交集面积是否大于行字符包围盒面积的80%
                    if intersection_area >= 0.6 * line_area:
                        line_shape['region_label'] = region_shape['label']
                        break
                    else:
                        line_shape['region_label'] = None
            
            # 将结果写入新的JSON文件
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w') as file:
                json.dump(line_data, file, indent=4)
            print(f"Processed {filename}")

class Line22Line1():
    def __init__(self):
        pass
    def bezier_curve(self, p0, p1, p2, p3, t):
        """计算贝塞尔曲线在t点的值"""
        x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
        return x, y

    def calculate_center_line(self, upper_curve, lower_curve, num_samples=100):
        """计算贝塞尔曲线的中心线"""
        upper_points = [self.bezier_curve(*upper_curve, t) for t in np.linspace(0, 1, num_samples)]
        lower_points = [self.bezier_curve(*lower_curve, t) for t in np.linspace(0, 1, num_samples)]
        center_points = [( (up[0] + lp[0]) / 2, (up[1] + lp[1]) / 2) for up, lp in zip(upper_points, lower_points)]
        return center_points

    def is_point_in_bbox(self, point, bbox):
        """判断点是否在包围盒（bbox）中"""
        x, y = point
        (x_min, y_min), (x_max, y_max) = bbox
        return x_min <= x <= x_max and y_min <= y <= y_max

    def is_point_in_bbox_other(self, point, bbox):
        """判断点是否在包围盒（bbox）中"""
        x, y = point
        x_min, y_min, x_max, y_max = bbox
        return x_min <= x <= x_max and y_min <= y <= y_max

    def percentage_points_in_bbox(self, points, bbox):
        """计算中心线采样点在包围盒中的百分比"""
        if len(bbox) == 2:
            count = sum(1 for point in points if self.is_point_in_bbox(point, bbox))
        elif len(bbox) == 4:
            count = sum(1 for point in points if self.is_point_in_bbox_other(point, bbox))
        return count / len(points)

    def process(self, bezier_curve_folder, bbox_folder, output_folder, filename):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历第一个文件夹
        # for filename in os.listdir(bezier_curve_folder):
        if filename.endswith('.json'):
            bezier_curve_path = os.path.join(bezier_curve_folder, filename)
            bbox_path = os.path.join(bbox_folder, filename)
            if not os.path.exists(bbox_path):
                print(f"Failed Processed {filename}")
                # continue
                return
            # 读取贝塞尔曲线JSON文件
            with open(bezier_curve_path, 'r') as file:
                bezier_data = json.load(file)
            
            # 读取包围盒JSON文件
            with open(bbox_path, 'r') as file:
                bbox_data = json.load(file)
            
            # 遍历每个贝塞尔曲线
            for bezier_shape in bezier_data['shapes']:
                bezier_points = bezier_shape['points']
                
                # 上下两条贝塞尔曲线的控制点
                upper_curve = [(bezier_points[i], bezier_points[i + 1]) for i in range(0, 8, 2)]
                lower_curve = [(bezier_points[i], bezier_points[i + 1]) for i in range(8, 16, 2)]
                
                # 计算贝塞尔曲线的中心线
                center_line = self.calculate_center_line(upper_curve, lower_curve)
                
                # 遍历每个包围盒
                for bbox_shape in bbox_data['shapes']:
                    bbox_points = bbox_shape['points']
                    
                    # 计算采样点在包围盒中的百分比
                    percentage = self.percentage_points_in_bbox(center_line, bbox_points)
                    
                    # 判断中心线的采样点是否有60%以上在包围盒中
                    if percentage >= 0.6:
                        bezier_shape['bbox_label'] = bbox_shape['label']
                        break
                    else:
                        bezier_shape['bbox_label'] = None
            
            # 将结果写入新的JSON文件
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w') as file:
                json.dump(bezier_data, file, indent=4)
            print(f"Processed {filename}")

def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)

def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs

class DataRender():
    def __init__(self):
        self.font_size = 12

        self.char_color = (255, 255, 0)
        self.char_alpha = 0.8
        self.char_color = (255, 0, 0)
        self.char_alpha = 0.8
        self.line_color = (0, 255, 0)
        self.line_alpha = 0.8
        self.block_color = (0, 0, 255)
        self.block_alpha = 0.8

        self.delta = 0.4
        self.draw_map = {
            'block_bbox': [self.draw_bbox, self.block_color, self.block_alpha],
            'line_bbox': [self.draw_bbox, self.line_color, self.line_alpha],
            'char_bezier': [self.draw_bezier, self.char_color, self.char_alpha],
            # 'char_bezier': [self.draw_polygon, self.char_color, self.char_alpha]
        }

    def draw_polygon(self, input_img, data, color, alpha=0.3):
        """在图像上绘制多边形并绘制边界框"""
        if len(data) == 2:
            
            points, score = data
            assert points.shape[1] == 8, 'points shape must be (n, 8)'
            for i in range(points.shape[0]):
                if score[i].item() > 0.5:
                    alpha1 = alpha
                else:
                    alpha1 = alpha - self.delta
                overlay = Image.new("RGBA", input_img.size)
                draw = ImageDraw.Draw(overlay)
                polygon = points[i].view(-1, 2)
                poly_points = [(int(x), int(y)) for x, y in polygon]
                color_with_alpha = color + (int(alpha1 * 255),)
                draw.polygon(poly_points, outline=color+(int(255),), fill=color_with_alpha)
                draw.text((int(polygon[0][0]-self.font_size), int(polygon[0][1]-self.font_size)), str(round(score[i].item(), 2)), fill=color, fontsize=self.font_size)
                input_img = Image.alpha_composite(input_img, overlay)
            return input_img
        else:
            points = data[0]
            assert points.shape[1] == 16, 'points shape must be (n, 16)'
            for i in range(points.shape[0]):
                alpha1 = alpha

                overlay = Image.new("RGBA", input_img.size)
                draw = ImageDraw.Draw(overlay)
                polygon = points[i].view(-1, 2)
                poly_points = [(int(x), int(y)) for x, y in polygon]
                color_with_alpha = color + (int(alpha1 * 255),)
                draw.polygon(poly_points, outline=color+(int(255),), fill=color_with_alpha)
                # draw.text((int(polygon[0][0]-self.font_size), int(polygon[0][1]-self.font_size)), str(round(score[i].item(), 2)), fill=color, fontsize=self.font_size)
                input_img = Image.alpha_composite(input_img, overlay)
            return input_img

    def draw_bbox(self, input_img, data, color, alpha=0.3):
        """在图像上绘制包围盒并绘制边界框"""
        if len(data) == 2:
            points, score = data
            assert points.shape[1] == 4, 'points shape must be (n, 4)'
            for i in range(points.shape[0]):
                if score[i].item() > 0.5:
                    alpha1 = alpha
                else:
                    alpha1 = alpha - self.delta
                overlay = Image.new("RGBA", input_img.size)
                draw = ImageDraw.Draw(overlay)
                x1, y1, x2, y2 = points[i]
                color_with_alpha = color + (int(alpha1 * 255),)
                draw.rectangle([(x1, y1), (x2, y2)], outline=color+(int(255),), fill=color_with_alpha)
                draw.text((int(x1-self.font_size), int(y1-self.font_size)), str(round(score[i].item(), 2)), fill=color, fontsize=self.font_size)
                input_img = Image.alpha_composite(input_img, overlay)
            return input_img
        else:
            points = data[0]
            assert points.shape[1] == 4, 'points shape must be (n, 4)'
            for i in range(points.shape[0]):
                alpha1 = alpha

                overlay = Image.new("RGBA", input_img.size)
                draw = ImageDraw.Draw(overlay)
                x1, y1, x2, y2 = points[i]
                # box_size = 10
                color_with_alpha = color + (int(alpha1 * 255),)
                draw.rectangle([(x1, y1), (x2, y2)], outline=color+(int(255),), fill=color_with_alpha)
                # draw.text((int(x1-self.font_size), int(y1-self.font_size)), fill=color, fontsize=self.font_size)
                input_img = Image.alpha_composite(input_img, overlay)
            return input_img
    
    def draw_bezier(self, input_img, data, color, alpha=0.3):
        """在图像上绘制贝塞尔曲线"""
        if len(data) == 2:
            points, score = data
            assert points.shape[1] == 16, 'points shape must be (n, 16)'
            for i in range(points.shape[0]):
                if score[i].item() > 0.5:
                    alpha1 = alpha
                else:
                    alpha1 = alpha - self.delta
                overlay = Image.new("RGBA", input_img.size)
                draw = ImageDraw.Draw(overlay)
                bezier_points = points[i]
                color_with_alpha = color + (int(alpha1 * 255),)
                # 上方曲线的四个控制点
                p0 = (int(bezier_points[0]), int(bezier_points[1]))
                p1 = (int(bezier_points[2]), int(bezier_points[3]))
                p2 = (int(bezier_points[4]), int(bezier_points[5]))
                p3 = (int(bezier_points[6]), int(bezier_points[7]))
                prev_point = p0
                for t in np.linspace(0, 1, 100):
                    x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
                    y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
                    current_point = (int(x), int(y))
                    draw.line([prev_point, current_point], fill=color_with_alpha, width=10)
                    prev_point = current_point
                draw.text((int(p0[0]-self.font_size), int(p0[1]-self.font_size)), str(round(score[i].item(), 2)), fill=color, fontsize=self.font_size)

                # 下方曲线的四个控制点
                p0 = (int(bezier_points[8]), int(bezier_points[9]))
                p1 = (int(bezier_points[10]), int(bezier_points[11]))
                p2 = (int(bezier_points[12]), int(bezier_points[13]))
                p3 = (int(bezier_points[14]), int(bezier_points[15]))
                prev_point = p0
                for t in np.linspace(0, 1, 100):
                    x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
                    y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
                    current_point = (int(x), int(y))
                    draw.line([prev_point, current_point], fill=color_with_alpha, width=10)
                    prev_point = current_point
                input_img = Image.alpha_composite(input_img, overlay)
            return input_img
        else:
            points = data[0]
            assert points.shape[1] == 16, 'points shape must be (n, 16)'
            for i in range(points.shape[0]):
                alpha1 = alpha
                overlay = Image.new("RGBA", input_img.size)
                draw = ImageDraw.Draw(overlay)
                bezier_points = points[i]
                color_with_alpha = color + (int(alpha1 * 255),)
                # 上方曲线的四个控制点
                p0 = (int(bezier_points[0]), int(bezier_points[1]))
                p1 = (int(bezier_points[2]), int(bezier_points[3]))
                p2 = (int(bezier_points[4]), int(bezier_points[5]))
                p3 = (int(bezier_points[6]), int(bezier_points[7]))
                prev_point = p0
                for t in np.linspace(0, 1, 100):
                    x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
                    y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
                    current_point = (int(x), int(y))
                    draw.line([prev_point, current_point], fill=color_with_alpha, width=10)
                    prev_point = current_point
                # draw.text((int(p0[0]-self.font_size), int(p0[1]-self.font_size)), str(round(score[i].item(), 2)), fill=color, fontsize=self.font_size)

                # 下方曲线的四个控制点
                p0 = (int(bezier_points[8]), int(bezier_points[9]))
                p1 = (int(bezier_points[10]), int(bezier_points[11]))
                p2 = (int(bezier_points[12]), int(bezier_points[13]))
                p3 = (int(bezier_points[14]), int(bezier_points[15]))
                prev_point = p0
                for t in np.linspace(0, 1, 100):
                    x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
                    y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
                    current_point = (int(x), int(y))
                    draw.line([prev_point, current_point], fill=color_with_alpha, width=10)
                    prev_point = current_point
                input_img = Image.alpha_composite(input_img, overlay)
            return input_img
    
    def render(self, render_res):
        ''' render all level sperated results'''
        res = {
            'input_imgs': [],
            'block_bbox': [],
            'line_bbox': [],
            'char_bezier': [],
        }
        for img_item in render_res:
            input_img = Image.open(img_item['img_path'])
            input_img = input_img.convert('RGBA')
            res['input_imgs'].append(input_img)

            for k, v in img_item['results'].items():
                output_img = self.draw_map[k][0](input_img, v, self.draw_map[k][1], self.draw_map[k][2])
                res[k].append(output_img)
        return res

    def pil_to_nchw(self, pil_list):
        """
        将 RGBA PIL 图像列表转换为 (N, 3, H, W) 的 numpy 数组
        参数:
            pil_list: List[PIL.Image] (RGBA 格式)
        返回:
            np.ndarray: (N, 3, H, W), dtype=uint8
        """
        np_arrays = []
        for pil_img in pil_list:
            rgb_img = pil_img.convert("RGB")
            np_img = np.array(rgb_img)
            np_img = np.transpose(np_img, (2, 0, 1))
            np_arrays.append(np_img)
        
        return np.stack(np_arrays, axis=0)

class ShowLabelInference():
    def __init__(self):
        pass
    def draw_polygon(self, draw, points, color, alpha, score):
        """在图像上绘制多边形并绘制边界框"""
        points = [(int(x), int(y)) for x, y in points]
        color_with_alpha = color + (int(alpha * 255),)
        draw.polygon(points, outline=color+(int(255),), fill=color_with_alpha)
        # draw.text(())
        # draw.line(points + [points[0]], fill=color_with_alpha, width=1)

    def draw_bbox(self, draw, points, color, alpha, score):
        """在图像上绘制包围盒并绘制边界框"""
        points = [(int(x), int(y)) for x, y in points]
        color_with_alpha = color + (int(alpha * 255),)
        draw.rectangle([points[0], points[1]], outline=color+(int(255),), fill=color_with_alpha)
        # draw.text((points[0][0], points[0][1]-10), str(score), fill=color+(int(255),))
        # draw.rectangle([points[0], points[1]], outline=color_with_alpha, width=1)

    def draw_bbox_other(self, draw, points, color, alpha, score):
        """在图像上绘制包围盒并绘制边界框"""
        points = [(int(points[0]), int(points[1])), (int(points[2]), int(points[3]))]
        color_with_alpha = color + (int(alpha * 255),)
        draw.rectangle([points[0], points[1]], outline=color+(int(255),), fill=color_with_alpha)
        draw.text((points[0][0], points[0][1]-12), str(score), fill=color+(int(255),), fontsize=12)

    def draw_bezier(self, draw, points, color, alpha, score):
        """在图像上绘制贝塞尔曲线"""
        color_with_alpha = color + (int(alpha * 255),)
        if len(points) == 16:
            # 上方曲线的四个控制点
            p0_upper = (int(points[0]), int(points[1]))
            p1_upper = (int(points[2]), int(points[3]))
            p2_upper = (int(points[4]), int(points[5]))
            p3_upper = (int(points[6]), int(points[7]))
            # prev_point = p0
            # for t in np.linspace(0, 1, 100):
            #     x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
            #     y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
            #     current_point = (int(x), int(y))
            #     draw.line([prev_point, current_point], fill=color_with_alpha, width=10)
            #     prev_point = current_point
            # draw.text((p0[0], p0[1]-12), str(score), fill=color_with_alpha, fontsize=12)

            # 下方曲线的四个控制点
            p0_lower = (int(points[8]), int(points[9]))
            p1_lower = (int(points[10]), int(points[11]))
            p2_lower = (int(points[12]), int(points[13]))
            p3_lower = (int(points[14]), int(points[15]))
            # prev_point = p0
            # for t in np.linspace(0, 1, 100):
            #     x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
            #     y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
            #     current_point = (int(x), int(y))
            #     draw.line([prev_point, current_point], fill=color_with_alpha, width=10)
            #     prev_point = current_point
            # 采样上方和下方曲线的点
            upper_points = [self.bezier_curve(p0_upper, p1_upper, p2_upper, p3_upper, t) for t in np.linspace(0, 1, 100)]
            lower_points = [self.bezier_curve(p0_lower, p1_lower, p2_lower, p3_lower, t) for t in np.linspace(0, 1, 100)]

            # 反转下方曲线的点顺序
            # lower_points.reverse()

            # 组合上方和下方的点，形成一个多边形
            polygon_points = upper_points + lower_points

            # 绘制多边形
            draw.polygon(polygon_points, outline=color_with_alpha, fill=color_with_alpha)

    
    def bezier_curve(self, p0, p1, p2, p3, t):
        """计算贝塞尔曲线在 t 点的值"""
        x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
        return x, y

    def bezier_length(self, p0, p1, p2, p3, num_samples=100):
        """计算贝塞尔曲线的长度"""
        length = 0
        prev_point = p0
        for t in np.linspace(0, 1, num_samples):
            current_point = self.bezier_curve(p0, p1, p2, p3, t)
            length += np.linalg.norm(np.array(current_point) - np.array(prev_point))
            prev_point = current_point
        return length

    def bezier_sample(self, p0, p1, p2, p3, num_segments=10):
        """根据贝塞尔曲线的长度均匀分布采样点"""
        length = self.bezier_length(p0, p1, p2, p3)
        segment_length = length / num_segments
        points = [p0]
        prev_point = p0
        accumulated_length = 0
        
        for t in np.linspace(0, 1, 1000):
            current_point = self.bezier_curve(p0, p1, p2, p3, t)
            accumulated_length += np.linalg.norm(np.array(current_point) - np.array(prev_point))
            
            if accumulated_length >= segment_length:
                points.append(current_point)
                accumulated_length = 0
            
            prev_point = current_point
        
        if len(points) < num_segments + 1:
            points.append(p3)
        
        return points

    def offset_point(self, p1, p2, distance):
        """沿着p1到p2的方向，缩短distance距离"""
        vec = np.array(p2) - np.array(p1)
        vec_length = np.linalg.norm(vec)
        if vec_length == 0:
            return p1
        offset_vec = vec / vec_length * distance
        return tuple(np.array(p1) + offset_vec)

    def get_polygon_angle(self, polygon):
        """计算多边形的主方向角度"""
        p1, p2 = polygon[0], polygon[1]
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        return angle

    def draw_text_in_polygon(self, draw, polygon, text, font_path):
        """在多边形内绘制文本，使其大小与多边形基本匹配，并且方向一致"""

        min_x = min(p[0] for p in polygon)
        max_x = max(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_y = max(p[1] for p in polygon)
        
        width = max_x - min_x
        height = max_y - min_y
        
        font_size = width
        font = ImageFont.truetype(font_path, font_size)
        
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # text_x = min_x + (width - text_width) / 2
        text_y = min_y + (height - text_height) / 2
        text_x = min_x
        # text_y = min_y
        
        # 计算多边形的方向角度
        angle = self.get_polygon_angle(polygon)
        
        # 创建旋转的文本图像
        text_img = Image.new("RGBA", (width+2, height+2), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((0, 0), text, font=font, fill=(0, 0, 0))
        
        # 旋转文本图像
        rotated_text_img = text_img.rotate(-angle, expand=True)
        
        # 将旋转后的文本粘贴到原始图像上
        draw.bitmap((text_x, text_y), rotated_text_img, fill=(0, 0, 0, 255))

    def draw_bezier_as_polygons(self, draw, points, color, alpha, score, text, font_path, num_segments=10, spacing=2, render_text=True):
        """在图像上绘制由贝塞尔曲线分割出的多边形，且多边形之间间隔2个像素"""
        color_with_alpha = color + (int(alpha * 255),)
        if text is not None:
            num_segments = len(text)
        if len(points) == 16:
            # 上方曲线的四个控制点
            p0_upper = (points[0], points[1])
            p1_upper = (points[2], points[3])
            p2_upper = (points[4], points[5])
            p3_upper = (points[6], points[7])

            # 下方曲线的四个控制点
            p0_lower = (points[8], points[9])
            p1_lower = (points[10], points[11])
            p2_lower = (points[12], points[13])
            p3_lower = (points[14], points[15])

            # 获取上方和下方曲线的采样点（均匀分布）
            upper_points = self.bezier_sample(p0_upper, p1_upper, p2_upper, p3_upper, num_segments)
            lower_points = self.bezier_sample(p0_lower, p1_lower, p2_lower, p3_lower, num_segments)

            # 反转下方曲线的点顺序
            lower_points.reverse()

            # 绘制多边形，增加间隔
            for i in range(num_segments):
                # 缩小多边形的大小，使得多边形之间有间隔
                upper_point1 = self.offset_point(upper_points[i], upper_points[i+1], spacing)
                upper_point2 = self.offset_point(upper_points[i+1], upper_points[i], spacing)
                lower_point1 = self.offset_point(lower_points[i], lower_points[i+1], spacing)
                lower_point2 = self.offset_point(lower_points[i+1], lower_points[i], spacing)

                polygon = [
                    (int(upper_point1[0]), int(upper_point1[1])),
                    (int(upper_point2[0]), int(upper_point2[1])),
                    (int(lower_point2[0]), int(lower_point2[1])),
                    (int(lower_point1[0]), int(lower_point1[1]))
                ]
                if not render_text:
                    draw.polygon(polygon, outline=color_with_alpha, fill=color_with_alpha)
                # 在多边形内绘制对应的文本
                if render_text and i < len(text):  # 确保字符数不超过多边形数
                    self.draw_text_in_polygon(draw, polygon, text[i], font_path)
            
            # 在起始点标记score
            # draw.text((int(p0_upper[0]), int(p0_upper[1])-12), str(score), fill=color_with_alpha)
    
    def process(self, output_folder, line_folder, char_folder, block_folder, image_folder, font_path, text = ""):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path, output_path_white, output_path_white_char = None, None, None
        # 遍历char文件夹
        for filename in os.listdir(line_folder):
            if filename.endswith('.json'):
                # char_path = os.path.join(char_folder, filename)
                block_path = os.path.join(block_folder, filename)
                line_path = os.path.join(line_folder, filename)
                char_path = os.path.join(char_folder, filename)
                if image_folder.split('/')[-1].split('.')[-1] in ['jpg', 'png']:
                    image_path = image_folder
                else:
                    image_path = os.path.join(image_folder, filename.replace('.json', '.jpg'))  # 假设图片格式为PNG

                if (not os.path.exists(line_path)) or (not os.path.exists(char_path)) or (not os.path.exists(block_path)) or (not os.path.exists(image_path)):
                    print(f"Skipping {filename} because some files are missing")
                    continue
                # 读取JSON文件
                # with open(char_path, 'r') as file:
                #     char_data = json.load(file)
                with open(line_path, 'r') as file:
                    line_data = json.load(file)
                with open(char_path, 'r') as file:
                    char_data = json.load(file)
                with open(block_path, 'r') as file:
                    block_data = json.load(file)

                # 读取背景图片
                image = Image.open(image_path).convert("RGBA")
                image_white = Image.new("RGB", image.size, (255, 255, 255)).convert("RGBA")
                image_white_char = Image.new("RGB", image.size, (255, 255, 255)).convert("RGBA")
                # overlay = Image.new("RGBA", image.size)
                # draw = ImageDraw.Draw(overlay)

                # 标记哪些char、line、block已经被绘制
                drawn_char = set()
                drawn_line = set()
                drawn_block = set()
                total = []
                change = False
                
                max_prob_char = {}
                max_prob_line = {}
                max_prob_block = {}
                # 绘制对应的char
                for char_shape in char_data['shapes']:
                    char_label = char_shape.get('label')
                    char_score = char_shape.get('score')
                    line_label = char_shape.get('bbox_label')
                    if line_label is not None and line_label not in drawn_line:
                        # 绘制对应的line
                        for line_shape in line_data['shapes']:
                            if line_shape['label'] == line_label:
                                line_score = line_shape.get('score')
                                block_label = line_shape.get('region_label')
                                if block_label is not None and block_label not in drawn_block:
                                    # 绘制对应的block
                                    for block_shape in block_data['shapes']:
                                        if block_shape['label'] == block_label:
                                            block_score = block_shape.get('score')
                                            overlay = Image.new("RGBA", image.size)
                                            draw = ImageDraw.Draw(overlay)

                                            if block_score > 0.85:
                                                # draw_bbox_other(draw, block_shape['points'], (0, 0, 255), alpha=0.3, score=block_shape['score'])  # 蓝色
                                                drawn_block.add(block_label)
                                                if line_score > 0.85:
                                                    # draw_bbox_other(draw, line_shape['points'], (0, 255, 0), alpha=0.4, score=line_shape['score'])  # 绿色
                                                    drawn_line.add(line_label)
                                                else:
                                                    if block_label not in max_prob_line:
                                                        max_prob_line[block_label] = line_shape
                                                    elif line_score > max_prob_line[block_label]['score']:
                                                        max_prob_line[block_label] = line_shape
                                                if line_label not in max_prob_char:
                                                    max_prob_char[line_label] = char_shape
                                                elif char_score > max_prob_char[line_label]['score']:
                                                    max_prob_char[line_label] = char_shape
                                            else:
                                                if 0 not in max_prob_block:
                                                    max_prob_block[0] = block_shape
                                                elif block_score > max_prob_block[0]['score']:
                                                    max_prob_block[0] = block_shape
                                            image = Image.alpha_composite(image, overlay)
                                            change = True   
                                elif block_label is not None and block_label in drawn_block:
                                    overlay = Image.new("RGBA", image.size)
                                    draw = ImageDraw.Draw(overlay)   

                                    if line_score > 0.85:
                                        # draw_bbox_other(draw, line_shape['points'], (0, 255, 0), alpha=0.4, score=line_shape['score'])  # 绿色
                                        drawn_line.add(line_label)
                                    else:
                                        if block_label not in max_prob_line:
                                            max_prob_line[block_label] = line_shape
                                        elif line_score > max_prob_line[block_label]['score']:
                                            max_prob_line[block_label] = line_shape
                                    if line_label not in max_prob_char:
                                        max_prob_char[line_label] = char_shape
                                    elif char_score > max_prob_char[line_label]['score']:
                                        max_prob_char[line_label] = char_shape
                                    image = Image.alpha_composite(image, overlay)
                                    change = True
                    elif line_label is not None and line_label in drawn_line:
                        overlay = Image.new("RGBA", image.size)
                        draw = ImageDraw.Draw(overlay) 

                        if line_label not in max_prob_char:
                            max_prob_char[line_label] = char_shape
                        elif char_score > max_prob_char[line_label]['score']:
                            max_prob_char[line_label] = char_shape

                        image = Image.alpha_composite(image, overlay)
                        change = True
                for block_label, char_shape in max_prob_char.items():
                    overlay = Image.new("RGBA", image.size)
                    overlay_white = Image.new("RGBA", image.size)
                    overlay_white_char = Image.new("RGBA", image.size)

                    draw = ImageDraw.Draw(overlay)
                    draw_white = ImageDraw.Draw(overlay_white)
                    draw_white_char = ImageDraw.Draw(overlay_white_char)

                    self.draw_bezier_as_polygons(draw, char_shape['points'], (255, 0, 0), alpha=0.4, score=char_shape['score'], text=text, font_path=font_path, num_segments=8, spacing=2)  # 红色
                    # self.draw_bezier_as_polygons(draw_white, char_shape['points'], (0, 0, 0), alpha=1.0, score=char_shape['score'], text=text, font_path=font_path, num_segments=8, spacing=2, render_text=False)  # 红色
                    self.draw_bezier(draw_white, char_shape['points'], (0, 0, 0), alpha=1.0, score=char_shape['score'])  # 红色
                    self.draw_bezier_as_polygons(draw_white_char, char_shape['points'], (0, 0, 0), alpha=0.4, score=char_shape['score'], text=text, font_path=font_path, num_segments=8, spacing=2, render_text=True)  # 红色
                    

                    image = Image.alpha_composite(image, overlay)
                    image_white = Image.alpha_composite(image_white, overlay_white)
                    image_white_char = Image.alpha_composite(image_white_char, overlay_white_char)
                    change = True
                # 保存渲染后的图像
                if change:
                    output_path = os.path.join(output_folder, filename.replace('.json', '.png'))
                    output_path_white = os.path.join(output_folder, filename.replace('.json', '_white.png'))
                    output_path_white_char = os.path.join(output_folder, filename.replace('.json', '_white_char.png'))
                    image.save(output_path)
                    image_white.save(output_path_white)
                    image_white_char.save(output_path_white_char)
                    print(f"Saved image:\n {output_path}\n{output_path_white}\n{output_path_white_char}")
                else:
                    print(f"No changes for image: {filename}")
        return output_path, output_path_white, output_path_white_char
