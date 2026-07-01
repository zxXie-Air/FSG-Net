# FSG-Net

Official PyTorch implementation of **FSG-Net: Frequency-Spatial Synergistic Gated Network for High-Resolution Remote Sensing Change Detection**.

FSG-Net is a Siamese change detection network for bi-temporal remote sensing images. It combines frequency-domain wavelet interaction, temporal-spatial attention, and gated multi-scale feature fusion.

## Environment

The project was prepared for a CUDA 12 PyTorch environment. A recommended setup is:

```bash
conda create -n fsgnet python=3.8 -y
conda activate fsgnet
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt --no-deps
```

The `requirements.txt` file also records the PyTorch-side package versions used during development. For CUDA-enabled installations, installing PyTorch through conda first is recommended, then using `pip install -r requirements.txt --no-deps` to install the remaining Python packages without replacing the CUDA build.

If you prefer a pure pip setup, install the matching CUDA wheel from the official PyTorch instructions, then run:

```bash
pip install -r requirements.txt
```

## Dataset Format

Prepare each dataset with the following directory structure:

```text
dataset_root/
  train/
    A/
    B/
    label/
  val/
    A/
    B/
    label/
  test/
    A/
    B/
    label/
```

`A` and `B` contain the bi-temporal images. `label` contains binary change masks with the same filenames as the images. Non-zero label values are treated as changed pixels.

## Training

```bash
python train.py \
  --dataset-dir data/your_dataset \
  --cuda 0 \
  --batch-size 32 \
  --epochs 100 \
  --input-size 256
```

Training outputs are saved under:

```text
runs/<dataset_name>/train/<run_id>/
```

The best checkpoints are saved in the `weights/` subdirectory.

## Evaluation

```bash
python eval.py \
  --dataset-dir data/your_dataset \
  --ckp-paths runs/your_dataset/train/1/weights/epoch_x_x.pt \
  --cuda 0 \
  --batch-size 16 \
  --input-size 256
```

Evaluation writes metrics to `eval_metrics.csv` and saves four-color prediction maps under:

```text
runs/<dataset_name>/eval/<checkpoint_name>/predictions/
```

Color meaning:

- White: true positive
- Black: true negative
- Red: false positive
- Green: false negative

## Main Files

- `train.py`: training entry point
- `eval.py`: evaluation entry point
- `models/main_model.py`: FSG-Net model definition
- `models/my_block/dawim.py`: discrepancy-aware wavelet interaction module
- `models/my_block/stsam.py`: synergistic temporal-spatial attention module
- `models/my_block/lgfu.py`: lightweight gated fusion unit
- `utils/dataloaders.py`: dataset loading
- `losses/`: training losses

## Notes

- Model checkpoints and experiment outputs are not included in this repository.
- Dataset paths in the commands above are examples; replace them with your local dataset path.
- The default backbone is `resnet18` from `timm`.

## Citation

If this code is useful for your research, please cite the corresponding paper.
