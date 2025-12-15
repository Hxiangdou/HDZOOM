import os
import json
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
# from bezierSplit import draw_bezier_as_polygons

def bezier_curve(p0, p1, p2, p3, t):
    """计算贝塞尔曲线在 t 点的值"""
    x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]
    y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def draw_polygon(draw, points, color, alpha, score):
    """在图像上绘制多边形并绘制边界框"""
    points = [(int(x), int(y)) for x, y in points]
    color_with_alpha = color + (int(alpha * 255),)
    draw.polygon(points, outline=color+(int(255),), fill=color_with_alpha)
    # draw.text(())
    # draw.line(points + [points[0]], fill=color_with_alpha, width=1)

def draw_bbox(draw, points, color, alpha, score):
    """在图像上绘制包围盒并绘制边界框"""
    points = [(int(x), int(y)) for x, y in points]
    color_with_alpha = color + (int(alpha * 255),)
    draw.rectangle([points[0], points[1]], outline=color+(int(255),), fill=color_with_alpha)
    # draw.text((points[0][0], points[0][1]-10), str(score), fill=color+(int(255),))
    # draw.rectangle([points[0], points[1]], outline=color_with_alpha, width=1)

def draw_bbox_other(draw, points, color, alpha, score):
    """在图像上绘制包围盒并绘制边界框"""
    points = [(int(points[0]), int(points[1])), (int(points[2]), int(points[3]))]
    color_with_alpha = color + (int(alpha * 255),)
    draw.rectangle([points[0], points[1]], outline=color+(int(255),), fill=color_with_alpha)

def draw_bezier(draw, points, color, alpha, score):
    """在图像上绘制贝塞尔曲线"""
    color_with_alpha = color + (int(alpha * 255),)
    if len(points) == 16:
        # 上方曲线的四个控制点
        p0_upper = (int(points[0]), int(points[1]))
        p1_upper = (int(points[2]), int(points[3]))
        p2_upper = (int(points[4]), int(points[5]))
        p3_upper = (int(points[6]), int(points[7]))


        # 下方曲线的四个控制点
        p0_lower = (int(points[8]), int(points[9]))
        p1_lower = (int(points[10]), int(points[11]))
        p2_lower = (int(points[12]), int(points[13]))
        p3_lower = (int(points[14]), int(points[15]))

        # 采样上方和下方曲线的点
        upper_points = [bezier_curve(p0_upper, p1_upper, p2_upper, p3_upper, t) for t in np.linspace(0, 1, 100)]
        lower_points = [bezier_curve(p0_lower, p1_lower, p2_lower, p3_lower, t) for t in np.linspace(0, 1, 100)]

        # 反转下方曲线的点顺序
        # lower_points.reverse()

        # 组合上方和下方的点，形成一个多边形
        polygon_points = upper_points + lower_points

        # 绘制多边形
        draw.polygon(polygon_points, outline=color_with_alpha, fill=color_with_alpha)
# 文件夹路径
# char_folder = r"C:\Users\dell\Desktop\LATEX-EnxText-b\jsons\char2line2"

def render_generate_hdlayout(res_files, image_folder):

    line1_folder = os.path.join(res_files, "jsons/line1_bbox")
    line2_folder = os.path.join(res_files, "jsons/line2_bezier")
    block_folder = os.path.join(res_files, "jsons/block_bbox")
    output_folder = os.path.join(res_files, "masks")
    
    if len(image_folder.split('.')) > 1 :
        image_folder =os.path.split(image_folder)[0]
    
    font_path = r"./font/Arial_Unicode.ttf"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历char文件夹
    for filename in os.listdir(line1_folder):
        if filename.endswith('.json'):
            # char_path = os.path.join(char_folder, filename)
            block_path = os.path.join(block_folder, filename)
            line1_path = os.path.join(line1_folder, filename)
            line2_path = os.path.join(line2_folder, filename)
            image_path = os.path.join(image_folder, filename.replace('.json', '.jpg'))  # 假设图片格式为PNG

            if (not os.path.exists(line1_path)) or (not os.path.exists(line2_path)) or (not os.path.exists(block_path)) or (not os.path.exists(image_path)):
                print(f"Skipping {filename} because some files are missing")
                continue
            # 读取JSON文件
            # with open(char_path, 'r') as file:
            #     char_data = json.load(file)
            with open(line1_path, 'r') as file:
                line1_data = json.load(file)
            with open(line2_path, 'r') as file:
                line2_data = json.load(file)
            with open(block_path, 'r') as file:
                block_data = json.load(file)

            # 读取背景图片
            image = Image.open(image_path).convert("RGBA")
            image_white = Image.new("RGB", image.size, (255, 255, 255)).convert("RGBA")
            image_white_char = Image.new("RGB", image.size, (255, 255, 255)).convert("RGBA")
            # overlay = Image.new("RGBA", image.size)
            # draw = ImageDraw.Draw(overlay)

            # 标记哪些line2、line1、block已经被绘制
            drawn_line2 = set()
            drawn_line1 = set()
            drawn_block = set()
            total = []
            change = False
            
            max_prob_line2 = {}
            max_prob_line1 = {}
            max_prob_block = {}
            # 绘制对应的line2
            for line2_shape in line2_data['shapes']:
                line2_label = line2_shape.get('label')
                line2_score = line2_shape.get('score')
                line1_label = line2_shape.get('bbox_label')
                if line1_label is not None and line1_label not in drawn_line1:
                    # 绘制对应的line1
                    for line1_shape in line1_data['shapes']:
                        if line1_shape['label'] == line1_label:
                            line1_score = line1_shape.get('score')
                            block_label = line1_shape.get('region_label')
                            if block_label is not None and block_label not in drawn_block:
                                # 绘制对应的block
                                for block_shape in block_data['shapes']:
                                    if block_shape['label'] == block_label:
                                        block_score = block_shape.get('score')
                                        overlay = Image.new("RGBA", image.size)
                                        draw = ImageDraw.Draw(overlay)

                                        if block_score > 0.85:
                                            draw_bbox_other(draw, block_shape['points'], (0, 0, 255), alpha=0.4, score=block_shape['score'])  # 蓝色
                                            drawn_block.add(block_label)
                                            if line1_score > 0.85:
                                                draw_bbox_other(draw, line1_shape['points'], (0, 255, 0), alpha=0.4, score=line1_shape['score'])  # 绿色
                                                drawn_line1.add(line1_label)
                                            else:
                                                if block_label not in max_prob_line1:
                                                    max_prob_line1[block_label] = line1_shape
                                                elif line1_score > max_prob_line1[block_label]['score']:
                                                    max_prob_line1[block_label] = line1_shape
                                            if line1_label not in max_prob_line2:
                                                max_prob_line2[line1_label] = line2_shape
                                            elif line2_score > max_prob_line2[line1_label]['score']:
                                                max_prob_line2[line1_label] = line2_shape
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

                                if line1_score > 0.85:
                                    draw_bbox_other(draw, line1_shape['points'], (0, 255, 0), alpha=0.4, score=line1_shape['score'])  # 绿色
                                    drawn_line1.add(line1_label)
                                else:
                                    if block_label not in max_prob_line1:
                                        max_prob_line1[block_label] = line1_shape
                                    elif line1_score > max_prob_line1[block_label]['score']:
                                        max_prob_line1[block_label] = line1_shape
                                if line1_label not in max_prob_line2:
                                    max_prob_line2[line1_label] = line2_shape
                                elif line2_score > max_prob_line2[line1_label]['score']:
                                    max_prob_line2[line1_label] = line2_shape
                                image = Image.alpha_composite(image, overlay)
                                change = True
                elif line1_label is not None and line1_label in drawn_line1:
                    overlay = Image.new("RGBA", image.size)
                    draw = ImageDraw.Draw(overlay) 

                    if line1_label not in max_prob_line2:
                        max_prob_line2[line1_label] = line2_shape
                    elif line2_score > max_prob_line2[line1_label]['score']:
                        max_prob_line2[line1_label] = line2_shape

                    image = Image.alpha_composite(image, overlay)
                    change = True
            for block_label, block_shape in max_prob_block.items():
                if block_label not in drawn_block:
                    overlay = Image.new("RGBA", image.size)
                    draw = ImageDraw.Draw(overlay)   
                    draw_bbox_other(draw, block_shape['points'], (0, 0, 255), alpha=0.4, score=block_shape['score'])  # 蓝色
                    image = Image.alpha_composite(image, overlay)
                    drawn_block.add(0)
                    change = True
            for block_label, line1_shape in max_prob_line1.items():
                if line1_shape['label'] not in drawn_line1:
                    overlay = Image.new("RGBA", image.size)
                    draw = ImageDraw.Draw(overlay)
                    draw_bbox_other(draw, line1_shape['points'], (0, 255, 0), alpha=0.4, score=line1_shape['score'])  # 绿色
                    image = Image.alpha_composite(image, overlay)
                    drawn_line1.add(line1_shape['label'])
                    change = True
            for block_label, line2_shape in max_prob_line2.items():
                overlay = Image.new("RGBA", image.size)

                draw = ImageDraw.Draw(overlay)

                draw_bezier(draw, line2_shape['points'], (255, 0, 0), alpha=0.4, score=line2_shape['score'])

                image = Image.alpha_composite(image, overlay)

                change = True
            # 保存渲染后的图像
            if change:
                output_path = os.path.join(output_folder, filename.replace('.json', '.png'))

                image.save(output_path)

                print(f"Saved image: {output_path}")
            else:
                print(f"No changes for image: {filename}")

    print("Rendering completed and images saved.")
    
if __name__ == '__main__':
    
    res_files = r"/data/model/LATEX/20250325_163947/"
    image_folder = r"/data/LATEX-EnxText-new/val/images"
    
    render_generate_hdlayout(res_files, image_folder)
