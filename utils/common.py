import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def init_seed(seed=347):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from torch.backends import cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True


def check_dirs(dataset_name: str):
    print("\n" + "-" * 30 + "Check Dirs" + "-" * 30)

    runs_dir = Path("./runs") / dataset_name / "train"
    runs_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = [int(p.name) for p in runs_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    run_id = max(existing_runs) + 1 if existing_runs else 1

    save_path = runs_dir / str(run_id)
    weights_save_path = save_path / "weights"
    weights_save_path.mkdir(parents=True)

    print(f"Dataset: {dataset_name} | Run ID: {run_id}")
    print(f"Checkpoints & results will be saved at: {save_path}")

    result_save_path = save_path / "epoch_process.csv"
    best_ckp_file = None
    return str(save_path), str(weights_save_path), best_ckp_file, str(result_save_path)


def check_eval_dirs(dataset_name: str, ckp_path: str):
    print("\n" + "-" * 30 + "Check Dirs for Eval" + "-" * 30)

    base_eval_dir = Path("./runs") / dataset_name / "eval"
    base_eval_dir.mkdir(parents=True, exist_ok=True)

    p = Path(ckp_path)
    ckp_stem = p.stem
    train_run_id = None
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "train" and i + 1 < len(parts):
            train_run_id = parts[i + 1]
            break

    folder_name = f"train{train_run_id}_{ckp_stem}" if train_run_id else ckp_stem
    eval_save_path = base_eval_dir / folder_name
    eval_save_path.mkdir(exist_ok=True)

    print(f"Evaluating on dataset: {dataset_name}")
    print(f"Using checkpoint: {Path(ckp_path).name}")
    print(f"Evaluation results will be saved at: {eval_save_path}")

    result_csv_path = eval_save_path / "eval_metrics.csv"
    return str(eval_save_path), str(result_csv_path)


def compute_p_r_f1_miou_oa(tn_fp_fn_tps):
    p, r, f1, miou, oa = [], [], [], [], []
    for tn_fp_fn_tp in tn_fp_fn_tps:
        tn, fp, fn, tp = tn_fp_fn_tp
        p_tmp = tp / (tp + fp)
        r_tmp = tp / (tp + fn)
        iou = tp / (tp + fp + fn)
        oa_tmp = (tp + tn) / (tp + tn + fp + fn)

        p.append(p_tmp)
        r.append(r_tmp)
        f1.append(2 * p_tmp * r_tmp / (r_tmp + p_tmp))
        miou.append(iou)
        oa.append(oa_tmp)

    return np.array(p), np.array(r), np.array(f1), np.array(miou), np.array(oa)


def gpu_info():
    print("\n" + "-" * 30 + "GPU Info" + "-" * 30)
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        print(f"Using GPU count: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_mb = props.total_memory / 1024**2
            prefix = "Using CUDA " if i == 0 else " " * len("Using CUDA ")
            print(f"{prefix}device{i} name='{props.name}', memory={memory_mb:.0f}MB")
    else:
        print("Using CPU !!!")


class SaveResult:
    def __init__(self, result_save_path):
        self.result_save_path = result_save_path

    def prepare(self):
        header = "epoch,lr,P,R,F1,mIOU,OA,best_metric,train_loss,val_loss\n"
        with open(self.result_save_path, "w") as f:
            f.write(header)

    def show(self, p, r, f1, miou, oa, refer_metric=np.array(0), best_metric=0, train_avg_loss=0, val_avg_loss=0, lr=0, epoch=0):
        print(
            f"lr:{lr:.6f} | P:{p.mean():.5f} | R:{r.mean():.5f} | F1:{f1.mean():.5f} | "
            f"mIOU:{miou.mean():.5f} | OA:{oa.mean():.5f}\n"
            f"Current F1: {refer_metric.mean():.5f} | Best F1: {best_metric:.5f}"
        )

        log_data = (
            f"{epoch},{lr:.8f},{p.mean():.6f},{r.mean():.6f},{f1.mean():.6f},"
            f"{miou.mean():.6f},{oa.mean():.6f},{best_metric:.6f},"
            f"{train_avg_loss:.6f},{val_avg_loss:.6f}\n"
        )
        with open(self.result_save_path, "a") as f:
            f.write(log_data)


class ScaleInOutput:
    def __init__(self, input_size=512):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.output_size = None

    def scale_input(self, imgs: tuple):
        assert isinstance(imgs, tuple), "Please check the input type. It should be a 'tuple'."
        imgs = list(imgs)
        self.output_size = imgs[0].shape[2:]

        for i, img in enumerate(imgs):
            imgs[i] = F.interpolate(img, self.input_size, mode="bilinear", align_corners=True)

        return tuple(imgs)

    def scale_output(self, outs: tuple):
        if type(outs) in [torch.Tensor]:
            outs = (outs,)
        assert isinstance(outs, tuple), "Please check the input type. It should be a 'tuple'."
        outs = list(outs)

        assert self.output_size is not None, "Please call 'scale_input' function firstly."

        for i, out in enumerate(outs):
            outs[i] = F.interpolate(out, self.output_size, mode="bilinear", align_corners=True)

        return tuple(outs)
