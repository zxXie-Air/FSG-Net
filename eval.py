import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import platform
import cv2
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from models.main_model import ChangeDetection
from utils.dataloaders import get_eval_loaders
from utils.common import check_eval_dirs, compute_p_r_f1_miou_oa, gpu_info


def eval(opt):
    # ... (从设备设置到模型实例化的代码都保持不变) ...
    # 1. 设置设备
    if opt.cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU!!!")

    gpu_info()
    dataset_name = os.path.basename(opt.dataset_dir)
    print(f"Evaluating on Dataset: {dataset_name}")
    eval_save_path, result_save_path = check_eval_dirs(dataset_name, opt.ckp_paths[0])
    prediction_save_dir = Path(eval_save_path) / 'predictions'
    prediction_save_dir.mkdir(exist_ok=True)
    print(f"Prediction images will be saved to: {prediction_save_dir}")

    model_opt = argparse.Namespace(
        backbone='resnet18',  # 注意这里是 resnet18，与你报错日志一致
        neck='fpn+aspp+fuse+drop',
        head='fcn',
        dual_label=False,
        pretrain='',
        input_size=opt.input_size
    )

    print("Creating model instance...")
    model = ChangeDetection(opt=model_opt)

    # ... (前面的代码不变) ...

    # 2.3 加载权重文件
    ckp_path = opt.ckp_paths[0]
    print(f"-- Loading model checkpoint: {ckp_path}")
    checkpoint = torch.load(ckp_path, map_location=device)

    # ========================= 核心修复代码 (加强版) =========================

    # 步骤 A: 确定 state_dict 的来源
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 步骤 B: (加强版) 清洗 state_dict，过滤掉所有包含 'total_ops' 或 'total_params' 的键
    cleaned_state_dict = {
        k: v for k, v in state_dict.items()
        if 'total_ops' not in k and 'total_params' not in k
    }

    # 步骤 C: 处理 'module.' 前缀
    if cleaned_state_dict and list(cleaned_state_dict.keys())[0].startswith('module.'):
        cleaned_state_dict = {k[7:]: v for k, v in cleaned_state_dict.items()}

    # 步骤 D: 加载清洗后的权重
    # 使用 strict=False 可以增加加载的灵活性，但在这里我们先尝试严格加载
    # 如果仍然有问题，可以改为 model.load_state_dict(cleaned_state_dict, strict=False)
    model.load_state_dict(cleaned_state_dict)

    # =================================================================================

    print("Model weights loaded successfully.")

    # ... (后续代码不变) ...

    # 2.7 准备模型进行评估
    model.to(device)
    model.eval()

    # ... (后续的评估循环和结果保存代码完全不变) ...
    # 3. 获取数据加载器
    opt.dual_label = model_opt.dual_label
    eval_loader = get_eval_loaders(opt)

    # 4. 评估循环
    tn_fp_fn_tp = np.array([0, 0, 0, 0], dtype=np.int64)
    with torch.no_grad():
        eval_tbar = tqdm(eval_loader, desc="Evaluating")
        for batch_img1, batch_img2, batch_label, _, name in eval_tbar:
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            batch_label = batch_label.long().to(device)
            batch_label[batch_label != 0] = 1
            logits = model(batch_img1, batch_img2)
            cd_pred = torch.argmax(logits, dim=1)
            tn = ((cd_pred == 0) & (batch_label == 0)).sum().item()
            fp = ((cd_pred == 1) & (batch_label == 0)).sum().item()
            fn = ((cd_pred == 0) & (batch_label == 1)).sum().item()
            tp = ((cd_pred == 1) & (batch_label == 1)).sum().item()
            tn_fp_fn_tp += [tn, fp, fn, tp]
            for i in range(cd_pred.shape[0]):
                pred_map = cd_pred[i].cpu().numpy()
                gt_map = batch_label[i].squeeze().cpu().numpy()
                h, w = pred_map.shape
                color_map = np.zeros((h, w, 3), dtype=np.uint8)
                color_map[(pred_map == 1) & (gt_map == 1)] = [255, 255, 255]
                color_map[(pred_map == 0) & (gt_map == 0)] = [0, 0, 0]
                color_map[(pred_map == 1) & (gt_map == 0)] = [0, 0, 255]
                color_map[(pred_map == 0) & (gt_map == 1)] = [0, 255, 0]
                original_filename = Path(name[i])
                save_path = prediction_save_dir / original_filename.with_suffix('.png')
                cv2.imwrite(str(save_path), color_map)

    # 5. 计算并显示最终指标
    p, r, f1, miou, oa = compute_p_r_f1_miou_oa([tn_fp_fn_tp])
    print("\n" + "=" * 30 + " Results " + "=" * 30)
    print(f"F1: {f1.mean():.5f}")
    print(f"mIOU: {miou.mean():.5f}")
    print(f"Precision: {p.mean():.5f}")
    print(f"Recall: {r.mean():.5f}")
    print(f"OA: {oa.mean():.5f}")
    print("=" * 69)

    with open(result_save_path, 'w') as f:
        f.write("F1,mIOU,Precision,Recall,OA\n")
        f.write(f"{f1.mean()},{miou.mean()},{p.mean()},{r.mean()},{oa.mean()}\n")
    print(f"Evaluation metrics saved to {result_save_path}")
    print(f"F1-mean: {f1.mean():.5f}")
    print(f"mIOU-mean: {miou.mean():.5f}")
    print("=" * 69)


if __name__ == "__main__":
    # ... (这部分代码保持不变) ...
    parser = argparse.ArgumentParser('Change Detection eval')
    default_ckp_path = "runs/your_dataset/train/1/weights/epoch_x_x.pt"
    default_dataset_dir = 'data/your_dataset'
    parser.add_argument("--ckp-paths", type=list, default=[default_ckp_path], help="Path to the model checkpoint file.")
    parser.add_argument("--cuda", type=str, default="0", help="GPU ID")
    parser.add_argument("--dataset-dir", type=str, default=default_dataset_dir)
    parser.add_argument("--batch-size", type=int, default=16)
    default_num_workers = 0 if platform.system() == "Windows" else 8
    parser.add_argument("--num-workers", type=int, default=default_num_workers)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--tta", type=bool, default=False, help="Test Time Augmentation")
    opt = parser.parse_args()
    if not os.path.exists(opt.ckp_paths[0]):
        print(f"Error: Checkpoint path not found at '{opt.ckp_paths[0]}'")
        print("Please update the '--ckp-paths' argument or the default_ckp_path variable.")
    else:
        print("\n" + "-" * 30 + "OPT" + "-" * 30)
        # 从dataset_dir自动推断dataset_name，这是一个好习惯
        opt.dataset_name = Path(opt.dataset_dir).name
        print(opt)
        eval(opt)
