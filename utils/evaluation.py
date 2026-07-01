import numpy as np
import torch
from tqdm import tqdm
from .common import compute_p_r_f1_miou_oa, ScaleInOutput
# 注意，这里的.common是相对导入，假设common.py和evaluation_utils.py在同一个utils文件夹下


def eval_for_metric(model, eval_loader, criterion=None, tta=False, input_size=512):
    avg_loss = 0.0
    scale = ScaleInOutput(input_size)
    tn_fp_fn_tp = [np.array([0, 0, 0, 0], dtype=np.int64), np.array([0, 0, 0, 0], dtype=np.int64)]  # 使用64位整型

    model.eval()
    with torch.no_grad():
        # 健壮性检查
        if len(eval_loader) == 0:
            print("Warning: val_loader is empty. Returning zero metrics.")
            p, r, f1, miou, oa = compute_p_r_f1_miou_oa(tn_fp_fn_tp)
            return p.mean(), r.mean(), f1.mean(), miou.mean(), oa.mean(), 0.0

        eval_tbar = tqdm(eval_loader, desc="Validating", leave=False)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(eval_tbar):
            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            batch_label1 = batch_label1.long().cuda()
            batch_label2 = batch_label2.long().cuda()

            # ================== 关键修改 ==================
            batch_label1[batch_label1 != 0] = 1
            batch_label2[batch_label2 != 0] = 1
            # ===============================================

            if criterion is not None:
                batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))

            outs = model(batch_img1, batch_img2, tta)

            if not isinstance(outs, tuple):
                outs = (outs, outs)

            labels = (batch_label1, batch_label2)

            # 损失计算
            val_loss_item = 0.0
            if criterion is not None:
                outs_scaled = scale.scale_output(outs)
                loss = criterion(outs_scaled, labels)
                val_loss_item = loss.item()

            avg_loss = (avg_loss * i + val_loss_item) / (i + 1)
            eval_tbar.set_description(f"Validating... val_loss: {avg_loss:.4f}")

            # 预测图生成 (请确认模型输出形状！)
            # 假设输出是 (B, 2, H, W)
            cd_pred1 = torch.argmax(outs[0], 1)
            cd_pred2 = torch.argmax(outs[1], 1)
            cd_preds = (cd_pred1, cd_pred2)

            # 指标计算
            for j, (cd_pred, label) in enumerate(zip(cd_preds, labels)):
                tn = ((cd_pred == 0) & (label == 0)).sum().item()
                fp = ((cd_pred == 1) & (label == 0)).sum().item()
                fn = ((cd_pred == 0) & (label == 1)).sum().item()
                tp = ((cd_pred == 1) & (label == 1)).sum().item()

                # assert tn + fp + fn + tp == np.prod(label.shape) # 断言检查
                tn_fp_fn_tp[j] += [tn, fp, fn, tp]

    p, r, f1, miou, oa = compute_p_r_f1_miou_oa(tn_fp_fn_tp)

    return p.mean(), r.mean(), f1.mean(), miou.mean(), oa.mean(), avg_loss