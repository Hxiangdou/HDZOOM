class Config:
    model_path = "/home/fength/CreatiDesign/black-forest-labs/FLUX.1-dev" # 基础模型
    ckpt_repo = "HuiZhang0812/CreatiDesign"     # Checkpoint 仓库
    temp_path = "./temp"
    resolution = 1024 # 生成图像分辨率
    seed = 42
    num_inference_steps = 28
    guidance_scale = 1.0
    true_cfg_scale = 3.5
    
    my_condition_image_path = "/home/fength/CreatiDesign/T2I_test_data/14.jpg" 
    # mask_path = "/home/fength/CreatiDesign/test_data/infer-removebg-preview_mask.jpg"
    my_prompt = "Text:\"What Can I SAY\""
    
    # 布局[x1, y1, x2, y2] 坐标范围 0.0 到 1.0
    # my_layout = [
    #     {"bbox": [0.1, 0.1, 0.8, 0.2], "prompt": "Text:\"What Can I SAY\""},
    # ]
    
    output_dir = "outputs/custom_inference"

