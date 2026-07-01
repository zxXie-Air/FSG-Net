from tqdm import tqdm
import time
import random
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# --- 添加结束 ---
import torch
import argparse
from utils.evaluation import eval_for_metric
from losses.get_losses import SelectLoss
from models.block.Drop import dropblock_step
from utils.dataloaders import get_loaders
from utils.common import check_dirs, gpu_info, SaveResult, ScaleInOutput
from models.main_model import ChangeDetection

    
def downsample_label(label, new_size):
    batch_size, height, width = label.shape
    new_height, new_width = new_size
    stride_h = int(height / new_height)
    stride_w = int(width / new_width)
    downsampled_label = torch.zeros((batch_size, new_height, new_width), dtype=label.dtype, device=label.device)
    for i in range(0, height, stride_h):
        for j in range(0, width, stride_w):
            block = label[:, i:i+stride_h, j:j+stride_w]
            count_ones = block.sum(dim=[1, 2])
            count_zeros = stride_h * stride_w - count_ones

            downsampled_label[:, i//stride_h, j//stride_w] = (count_ones >= count_zeros).float()

    return downsampled_label


def train(opt):
    # init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    gpu_info()
    # 根据 opt.record 初始化日志记录功能
    if opt.record == 1:
        print("\n" + "=" * 20 + " RECORDING MODE ENABLED " + "=" * 20)
        # 当 record=1 时，创建实验文件夹并准备记录
        # ++++++++++++++ 修改开始 ++++++++++++++
        # 1. 从路径中提取数据集名称
        dataset_name = os.path.basename(opt.dataset_dir)
        print(f"Dataset Name: {dataset_name}")

        # 2. 将数据集名称传递给 check_dirs 函数
        save_path, best_ckp_save_path, best_ckp_file, result_save_path = check_dirs(dataset_name)
        #save_path, best_ckp_save_path, best_ckp_file, result_save_path = check_dirs()
        weights_save_path = best_ckp_save_path  # 将旧变量名赋给新变量名

        save_results = SaveResult(result_save_path)
        save_results.prepare()
        is_logging_enabled = True
    else:
        # 当 record=0 时，禁用所有文件操作
        print("\n" + "=" * 20 + " DEBUG MODE (NO LOGGING) " + "=" * 20)
        weights_save_path, best_ckp_file, save_results = None, None, None
        is_logging_enabled = False

    train_loader, val_loader = get_loaders(opt)
    scale = ScaleInOutput(opt.input_size)

    model = ChangeDetection(opt).cuda()
    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)
        model = torch.nn.DataParallel(model,device_ids = [0,1,2,3])
    criterion = SelectLoss(opt.loss)

    if opt.finetune:
        params = [{"params": [param for name, param in model.named_parameters()
                              if "backbone" in name], "lr": opt.learning_rate / 10},  
                  {"params": [param for name, param in model.named_parameters()
                              if "backbone" not in name], "lr": opt.learning_rate}]  
        print("Using finetune for model")
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)


    best_metric = 0
    train_avg_loss = 0
    total_bs = 32
    accumulate_iter = max(round(total_bs / opt.batch_size), 1)
    print("Accumulate_iter={} batch_size={}".format(accumulate_iter, opt.batch_size))

    for epoch in range(opt.epochs):
        model.train()
        train_tbar = tqdm(train_loader)
        for i, (batch_img1, batch_img2, batch_label, batch_label2, _) in enumerate(train_tbar):
            train_tbar.set_description("epoch {}, train_loss {}".format(epoch, train_avg_loss))
            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            batch_label = batch_label.long().cuda()
            batch_label[batch_label != 0] = 1
            batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))
            outs = model(batch_img1, batch_img2)
            outs = scale.scale_output(outs)
            loss = criterion(outs, (batch_label,))
            train_avg_loss = (train_avg_loss * i + loss.cpu().detach().numpy()) / (i + 1)
            loss.backward()
            if ((i+1) % accumulate_iter) == 0:
                optimizer.step()
                optimizer.zero_grad()
            del batch_img1, batch_img2, batch_label, batch_label2
        scheduler.step()
        dropblock_step(model)
        p, r, f1, miou, oa, val_avg_loss = eval_for_metric(model, val_loader, criterion, input_size=opt.input_size)
        refer_metric = f1  # 以F1作为参考指标
        underscore = "_"

        # 只有在记录模式下才保存权重和日志
        if is_logging_enabled:
            current_f1 = refer_metric.mean()
            if current_f1 > best_metric:
                # 1. 更新 best_metric 的值
                best_metric = current_f1
                print(f"\n---> New best F1: {best_metric:.5f}! Saving model... <---\n")

                # 2. 定义新模型的保存路径，文件名包含了epoch和F1分数
                simple_filename = f"epoch_{epoch}_{round(best_metric, 5)}.pt"

                new_best_ckp_file = os.path.join(weights_save_path, simple_filename)

                # 3. 保存模型权重，不再删除任何旧文件
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), new_best_ckp_file)

            lr = optimizer.state_dict()['param_groups'][0]['lr']
            # 调用 SaveResult.show 方法记录日志
            save_results.show(p, r, f1, miou, oa, refer_metric, best_metric, train_avg_loss, val_avg_loss, lr, epoch)
        else:
            # 在非记录模式下，只在控制台打印关键信息
            best_metric = max(best_metric, refer_metric.mean())  # 仍然需要更新best_metric用于显示
            print(
                f"Epoch {epoch} | Val_F1: {f1.mean():.5f} | Val_mIOU: {miou.mean():.5f} | Best_F1: {best_metric:.5f}"
            )

def set_randomness():
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection train')
    parser.add_argument("--record", type=int, default=1, help="0: no log, 1: log experiment")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")
    parser.add_argument("--pretrain", type=str,
                        default="")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--dataset-dir", type=str, default='data/your_dataset')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default= 0.001)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    # parser.add_argument("--pseudo-label", type=bool, default=True)
    start_time = time.time()
    opt = parser.parse_args()
    print(opt)
    set_randomness()
    train(opt)
    end_time = time.time()
    total_seconds = int(end_time - start_time)

    # 将总秒数转换为 时:分:秒 格式
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    print("\n" + "=" * 20 + " Training Finished " + "=" * 20)
    print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d} (H:M:S)")
    print("=" * 61)
