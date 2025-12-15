import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps
import os
# from ...CreatiDesign.infer_creatidesign_hdzoom import img_transforms, mask_transforms
def bezier_cubic_points(p0, p1, p2, p3, n_samples=20):
    """
    计算三次贝塞尔曲线上的采样点
    """
    t = np.linspace(0, 1, n_samples)
    points = []
    for val in t:
        # Bernstein polynomials
        x = (1-val)**3 * p0[0] + 3 * (1-val)**2 * val * p1[0] + \
            3 * (1-val) * val**2 * p2[0] + val**3 * p3[0]
        y = (1-val)**3 * p0[1] + 3 * (1-val)**2 * val * p1[1] + \
            3 * (1-val) * val**2 * p2[1] + val**3 * p3[1]
        points.append([x, y])
    return np.array(points, dtype=np.float32)

def process_outputs(img, outputs, target_size=(1024, 1024)):
    """
    处理模型输出数据，生成 Layout, Mask 和 Condition Image。

    Args:
        outputs (dict): 包含模型输出的字典，需包含:
                        'pred_line': 预测的行/框坐标 tensor [B, N, 4]
                        'pred_line_logits': 预测的行分类 logits [B, N, 2]
                        'pred_char': 预测的字符/区域 Bezier 控制点 tensor [B, N, 16]
                        'pred_char_logits': 预测的字符区域分类 logits [B, N, 2]
        target_size (tuple): 输出图像的分辨率 (W, H)

    Returns:
        dict: {
            "my_layout": list of dict, 布局数据 [{'bbox': [x1,y1,x2,y2], 'prompt': 'text_line'}, ...],
            "my_condition_img": PIL.Image, 经过 Mask 处理后的条件图像,
            "mask_img": PIL.Image, 生成的黑白掩码图,
            "visualization": PIL.Image, 可视化验证图 (原图+框+Mask叠加)
        }
    """
    
    W, H = target_size
    
    # 1. 提取并标准化数据
    # 假设 batch_size = 1，取第一个 batch
    # 确保数据在 CPU 上并转为 numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    # 处理输入图像 (img)
    if os.path.exists(img):
        img_tensor = Image.open(img).convert("RGB")
        original_img_pil = img_tensor.resize((W, H))
        original_img_cv = cv2.resize(np.array(img_tensor), (W, H))
        
    else:
        # 如果没有 img，创建全黑图占位
        original_img_pil = Image.new("RGB", (W, H), (0, 0, 0))
        original_img_cv = np.zeros((H, W, 3), dtype=np.uint8)

    pred_lines = to_numpy(outputs['pred_line'])
    pred_line_logits = to_numpy(outputs['pred_line_logits'])
    pred_chars = to_numpy(outputs['pred_char'])
    pred_char_logits = to_numpy(outputs['pred_char_logits'])

    # 如果是 Batch 维度 [1, N, ...], 去掉 Batch 维度
    if pred_lines.ndim == 3: pred_lines = pred_lines[0]
    if pred_line_logits.ndim == 3: pred_line_logits = pred_line_logits[0]
    if pred_chars.ndim == 3: pred_chars = pred_chars[0]
    if pred_char_logits.ndim == 3: pred_char_logits = pred_char_logits[0]

    # ================= 2. 处理 Layout (my_layout) =================
    # 筛选逻辑：pred_line_logits 的 Class 1 (index 1) > Class 0 (index 0)
    line_scores = pred_line_logits
    valid_line_indices = np.argmax(line_scores, axis=-1) == 1
    
    valid_lines = pred_lines[valid_line_indices]
    
    my_layout = []
    
    # 假设 pred_line 格式为 Transformer 常见的 [cx, cy, w, h] (中心坐标+宽高，归一化 0-1)
    # 也有可能是 [x1, y1, x2, y2]，根据 outputs.txt 中 [0.99, 0.20, 0.08, 0.02] 这种数据，
    # 极有可能是 [cx, cy, w, h] (0.99是中心靠右，0.08是宽度)
    # 或者 [x, y, w, h] (左上角+宽高)。这里先按 [cx, cy, w, h] 处理，如果效果不对可切换。
    
    for box in valid_lines:
        # box: [c_x, c_y, w, h]
        cx, cy, w, h = box
        
        # 转换为 [x1, y1, x2, y2]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # 截断到 [0, 1]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(1, x2), min(1, y2)
        
        my_layout.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "prompt": "text line" # 默认 prompt，因为输出只有位置信息
        })

    # ================= 3. 处理 Mask (pred_char Bezier) =================
    # 筛选逻辑：pred_char_logits 的 Class 1 > Class 0
    char_scores = pred_char_logits
    valid_char_indices = np.argmax(char_scores, axis=-1) == 1
    
    valid_chars = pred_chars[valid_char_indices] # [N, 16]
    
    # 创建一个空的 Mask 画布 (单通道)
    full_mask = np.zeros((H, W), dtype=np.uint8)
    
    for bezier_params in valid_chars:
        # bezier_params: 16 values -> 8 points (x, y)
        # 格式通常为：P1_x, P1_y, ... P8_x, P8_y
        # 前4个点 (8 values) 是上曲线 (Top Curve)
        # 后4个点 (8 values) 是下曲线 (Bottom Curve)
        points = bezier_params.reshape(8, 2)
        
        # 上曲线控制点 P0-P3
        top_curve_pts = points[0:4]
        # 下曲线控制点 P4-P7
        bottom_curve_pts = points[4:8]
        
        # 采样 Bezier 曲线点
        top_line = bezier_cubic_points(*top_curve_pts, n_samples=20)
        bottom_line = bezier_cubic_points(*bottom_curve_pts, n_samples=20)
        
        # 构建闭合多边形: Top(左->右) + Bottom(右->左)
        # 注意：需要检查点的顺序。通常 Text Spotting 中 Top 是从左到右，Bottom 也是从左到右。
        # 构成区域时，应该是 Top points + Reversed(Bottom points)
        # polygon_pts = np.vstack([top_line, bottom_line[::-1]])
        polygon_pts = np.vstack([top_line, bottom_line])
        
        # 归一化坐标 -> 像素坐标
        polygon_pts[:, 0] *= W
        polygon_pts[:, 1] *= H
        polygon_pts = polygon_pts.astype(np.int32)
        
        # 在 Mask 上填充白色
        cv2.fillPoly(full_mask, [polygon_pts], 255)

    mask_pil = Image.fromarray(full_mask, mode='L')

    # ================= 4. 生成条件图像 (my_condition_image) =================
    # myconditionimage = mask * img
    # 将 mask 转为 RGB 以便相乘
    mask_rgb = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2RGB) / 255.0 # [0, 1] float
    
    # 原始图像 (np array)
    img_float = original_img_cv.astype(np.float32)
    
    # 简单的像素乘法：Mask区域保留原图，背景变黑
    # 如果您需要灰底 (CreatiDesign 风格)，可以将背景设为 (128,128,128)
    masked_img_np = img_float * mask_rgb
    
    # 将背景(Mask为0的地方) 设为 灰色(128)
    background = np.ones_like(img_float) * 128.0
    # final_condition_np = masked_img_np + background * (1 - mask_rgb)
    final_condition_np = img_float * (1 - mask_rgb)  + background * mask_rgb
    
    my_condition_img = Image.fromarray(final_condition_np.astype(np.uint8))

    # ================= 5. 生成可视化验证图 =================
    vis_img = original_img_pil.copy()
    draw = ImageDraw.Draw(vis_img)
    
    # 画 Bbox (红色)
    for item in my_layout:
        bbox = item['bbox']
        # [x1, y1, x2, y2]
        rect = [
            bbox[0] * W, bbox[1] * H,
            bbox[2] * W, bbox[3] * H
        ]
        draw.rectangle(rect, outline="red", width=3)
    
    # 画 Mask (绿色半透明叠加)
    # 创建绿色图层
    green_layer = Image.new("RGB", (W, H), (0, 255, 0))
    # 使用 mask 作为 alpha 通道
    vis_img.paste(green_layer, (0, 0), mask=mask_pil)
    
    return {
        "my_layout": my_layout,
        "my_condition_img": my_condition_img,
        "mask_img": mask_pil,
        "visualization": vis_img
    }

def process_outputs_best_one(img, outputs, target_size=(1024, 1024), prompt="text line"):
    """
    处理模型输出，只选取置信度最高的 1 个 Bbox 和 1 个 Bezier 区域进行消除。
    """
    
    W, H = target_size
    
    # --- 1. 数据准备 (保持不变) ---
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    if os.path.exists(img):
        img_tensor = Image.open(img).convert("RGB")
        original_img_pil = img_tensor.resize((W, H))
        original_img_cv = cv2.resize(np.array(img_tensor), (W, H))
    else:
        original_img_pil = Image.new("RGB", (W, H), (0, 0, 0))
        original_img_cv = np.zeros((H, W, 3), dtype=np.uint8)

    pred_lines = to_numpy(outputs['pred_line'])           # [N, 4]
    pred_line_logits = to_numpy(outputs['pred_line_logits']) # [N, 2]
    pred_chars = to_numpy(outputs['pred_char'])           # [N, 16]
    pred_char_logits = to_numpy(outputs['pred_char_logits']) # [N, 2]

    # 去掉 Batch 维度 (如果有)
    if pred_lines.ndim == 3: pred_lines = pred_lines[0]
    if pred_line_logits.ndim == 3: pred_line_logits = pred_line_logits[0]
    if pred_chars.ndim == 3: pred_chars = pred_chars[0]
    if pred_char_logits.ndim == 3: pred_char_logits = pred_char_logits[0]

    # --- 2. 处理 Layout (选取 Logit 最高的 1 个框) ---
    
    # 获取所有预测对应 Class 1 的分数 (Logit)
    # pred_line_logits 形状通常为 [N, 2]，索引 1 是正类
    line_scores = pred_line_logits[:, 1]
    
    # 找到分数最高的索引
    best_line_idx = np.argmax(line_scores)
    
    # 提取最好的那个框
    best_box = pred_lines[best_line_idx]
    
    cx, cy, w, h = best_box
    x1, y1 = max(0, cx - w/2), max(0, cy - h/2)
    x2, y2 = min(1, cx + w/2), min(1, cy + h/2)
    
    my_layout = [{
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "prompt": prompt
    }]

    # --- 3. 处理 Mask (选取 Logit 最高的 1 个 Bezier) ---
    
    # 获取 Class 1 分数并找最大值索引
    char_scores = pred_char_logits[:, 1]
    best_char_idx = np.argmax(char_scores)
    
    # 提取最好的那组 Bezier 参数
    best_bezier_params = pred_chars[best_char_idx]
    
    # 绘制 Mask
    full_mask = np.zeros((H, W), dtype=np.uint8)
    
    points = best_bezier_params.reshape(8, 2)
    top_line = bezier_cubic_points(*points[0:4], n_samples=20)
    bottom_line = bezier_cubic_points(*points[4:8], n_samples=20)
    
    # 构成多边形
    polygon_pts = np.vstack([top_line, bottom_line])
    polygon_pts[:, 0] *= W
    polygon_pts[:, 1] *= H
    polygon_pts = polygon_pts.astype(np.int32)
    
    cv2.fillPoly(full_mask, [polygon_pts], 255)

    mask_pil = Image.fromarray(full_mask, mode='L')
    mask_pil = ImageOps.invert(mask_pil)
    # --- 4. 生成条件图像 (Mask 区域消除) ---
    mask_rgb = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2RGB) / 255.0 
    img_float = original_img_cv.astype(np.float32)
    erase_color = np.ones_like(img_float) * 128.0 # 灰色填充
    
    # 保留背景，Mask 区域变灰
    final_condition_np = img_float * (1 - mask_rgb) + erase_color * mask_rgb
    my_condition_img = Image.fromarray(final_condition_np.astype(np.uint8))

    # --- 5. 可视化验证 ---
    vis_img = original_img_pil.copy()
    draw = ImageDraw.Draw(vis_img)
    
    # 画红框 (现在只有 1 个)
    for item in my_layout:
        bbox = item['bbox']
        rect = [bbox[0]*W, bbox[1]*H, bbox[2]*W, bbox[3]*H]
        draw.rectangle(rect, outline="red", width=3)
    
    # 叠加红色遮罩
    red_layer = Image.new("RGB", (W, H), (255, 0, 0))
    vis_img.paste(red_layer, (0, 0), mask=mask_pil)
    
    return {
        "my_layout": my_layout,
        "my_condition_img": my_condition_img,
        "mask_img": mask_pil,
        "visualization": vis_img
    }
if __name__ == "__main__":
    
    print("Function 'process_outputs' is defined. Call it with your 'outputs' dictionary.")
    
    outputs = {
        'pred_block': [[[5.1111e-01, 5.2722e-02, 1.1804e-01, 7.8146e-03],
         [5.2878e-01, 1.2932e-02, 3.7198e-03, 7.2049e-05],
         [5.1914e-01, 1.8008e-02, 1.0346e-02, 2.4936e-04],
         [4.9705e-01, 4.8231e-01, 7.6839e-01, 5.8420e-01]]], 
        'pred_block_logits': [[[-7.3271,  6.8261],
         [-5.8413,  5.2595],
         [-7.2091,  6.7215],
         [-6.0811,  5.9374]]], 
       'pred_line': [[[0.9993, 0.2009, 0.0890, 0.0201],
         [0.9853, 0.3484, 0.0601, 0.0262],
         [0.7676, 0.1225, 0.0916, 0.0617],
         [0.3635, 0.4609, 0.6142, 0.3853],
         [0.4340, 0.5426, 0.6137, 0.3757],
         [0.3095, 0.3579, 0.4427, 0.3001],
         [0.8233, 0.2462, 0.1179, 0.0490],
         [0.4650, 0.8742, 0.4857, 0.4455],
         [0.6257, 0.4217, 0.5194, 0.3100],
         [0.5822, 0.0363, 0.0132, 0.0096],
         [0.2504, 0.0206, 0.0214, 0.0158],
         [0.4422, 0.4359, 0.6369, 0.4959],
         [0.2909, 0.4286, 0.4457, 0.3398],
         [0.6692, 0.0705, 0.0293, 0.0218],
         [0.9999, 0.5541, 0.0919, 0.0321],
         [0.2605, 0.1471, 0.1698, 0.1011]]], 
         'pred_line_logits': [[[ -7.2497,   8.5274],
         [ -7.4578,   8.6682],
         [-10.3683,   9.4309],
         [-11.1717,  10.9829],
         [ -7.0568,   8.9570],
         [-11.6497,  11.9075],
         [ -7.9123,   9.6886],
         [ -9.1330,   8.7459],
         [ -9.6789,   8.6641],
         [ -8.0484,   9.7412],
         [-10.9552,  13.2990],
         [ -7.7340,   8.8971],
         [ -9.4178,   9.7260],
         [ -9.5817,   9.9171],
         [ -6.8177,   7.9928],
         [-11.5234,  12.4086]]], 
       'pred_char': [[[0.1703, 0.5109, 0.2555, 0.4312, 0.2495, 0.4196, 0.3642, 0.4317,
          0.3660, 0.5228, 0.2531, 0.5083, 0.2687, 0.5346, 0.1963, 0.5941],
         [0.3605, 0.5524, 0.6269, 0.5441, 0.3685, 0.5521, 0.6407, 0.5348,
          0.6411, 0.6402, 0.3657, 0.6511, 0.6305, 0.6426, 0.3633, 0.6409],
         [0.3874, 0.2249, 0.6247, 0.2276, 0.4448, 0.2190, 0.6746, 0.2877,
          0.6549, 0.4357, 0.4326, 0.3734, 0.6250, 0.3802, 0.3940, 0.3664],
         [0.2091, 0.5637, 0.3190, 0.5500, 0.2778, 0.5511, 0.3969, 0.5641,
          0.4057, 0.6395, 0.2782, 0.6214, 0.3194, 0.6333, 0.2206, 0.6391],
         [0.1462, 0.5206, 0.2706, 0.2438, 0.6998, 0.2322, 0.8268, 0.5004,
          0.7628, 0.5722, 0.6340, 0.3376, 0.2961, 0.3693, 0.2094, 0.6103],
         [0.2258, 0.3923, 0.3853, 0.3065, 0.2654, 0.3292, 0.4261, 0.2794,
          0.4365, 0.4003, 0.2674, 0.4446, 0.3987, 0.4257, 0.2539, 0.4916],
         [0.3061, 0.5223, 0.3887, 0.4642, 0.4884, 0.4459, 0.5802, 0.5308,
          0.5471, 0.5877, 0.4605, 0.5021, 0.3872, 0.5463, 0.3260, 0.5755],
         [0.1616, 0.6902, 0.2910, 0.6942, 0.2017, 0.6978, 0.3384, 0.7158,
          0.3508, 0.7929, 0.2047, 0.7741, 0.2965, 0.7733, 0.1648, 0.7707],
         [0.3382, 0.5654, 0.5935, 0.5599, 0.3644, 0.5659, 0.6048, 0.5660,
          0.6112, 0.6836, 0.3581, 0.6700, 0.5797, 0.6668, 0.3399, 0.6771],
         [0.5486, 0.7926, 0.8353, 0.7884, 0.5809, 0.7906, 0.8606, 0.7907,
          0.8670, 0.8877, 0.5805, 0.8796, 0.8443, 0.8772, 0.5610, 0.8816],
         [0.3191, 0.8019, 0.8365, 0.7861, 0.3307, 0.8184, 0.8529, 0.7891,
          0.8704, 0.9068, 0.3336, 0.9116, 0.8459, 0.8965, 0.3221, 0.9103],
         [0.3924, 0.6767, 0.6677, 0.6565, 0.4151, 0.6730, 0.6870, 0.6486,
          0.6967, 0.7605, 0.4109, 0.7585, 0.6717, 0.7600, 0.3919, 0.7693],
         [0.2458, 0.4928, 0.7073, 0.4396, 0.2412, 0.4871, 0.7065, 0.4327,
          0.7143, 0.6399, 0.2463, 0.6879, 0.7127, 0.6454, 0.2450, 0.6976],
         [0.3671, 0.4398, 0.7830, 0.4404, 0.3815, 0.4493, 0.7913, 0.4494,
          0.7833, 0.6178, 0.3844, 0.6175, 0.7790, 0.6100, 0.3619, 0.6019],
         [0.3215, 0.7838, 0.4849, 0.7841, 0.3458, 0.7978, 0.5214, 0.7876,
          0.5471, 0.8727, 0.3718, 0.8798, 0.4883, 0.8668, 0.3345, 0.8698],
         [0.3564, 0.3649, 0.5805, 0.3304, 0.4148, 0.3204, 0.6407, 0.3456,
          0.6225, 0.4866, 0.4024, 0.4585, 0.5754, 0.4698, 0.3720, 0.4867]]], 
        'pred_char_logits': [[[-6.7080,  7.4305],
         [-7.6285,  7.9989],
         [-6.2280,  7.4100],
         [-6.0444,  6.7548],
         [-7.2050,  7.3821],
         [-7.2136,  8.6198],
         [-6.5913,  7.5042],
         [-6.8890,  7.6496],
         [-7.4599,  8.7470],
         [-5.9995,  6.4269],
         [-5.1033,  6.4163],
         [-8.2361,  8.8529],
         [-4.4913,  5.4618],
         [-7.7671,  8.7033],
         [-6.6524,  7.6745],
         [-6.4665,  7.4404]]]}
    
    img = "./test_data/14.jpg"  # 替换为实际图像路径或 tensor
    result = process_outputs(img=img, outputs=outputs)
    result['visualization'].save("./process_output_test/check_process.jpg")
    result['my_condition_img'].save("./process_output_test/condition.jpg")
    print(result['my_layout'])
    
    results = process_outputs_best_one(img=img, outputs=outputs)
    results['my_condition_img'].save("./process_output_test/condition_single_best.jpg")
    results['visualization'].save("./process_output_test/vis_single_best.jpg")
    results['mask_img'].save("./process_output_test/mask_single_best.jpg")
    print(results['my_layout'])