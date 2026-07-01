import os

import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from tqdm import tqdm


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if random.random() < 0.3:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            label1 = label1.transpose(Image.FLIP_LEFT_RIGHT)
            label2 = label2.transpose(Image.FLIP_LEFT_RIGHT)

        return img1, img2, label1, label2


class RandomCutout(object):
    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if random.random() < 0.5:
            img1 = self.cutout(img1)
            img2 = self.cutout(img2)
        return img1, img2, label1, label2

    def cutout(self, img):
        img = np.array(img)
        cut_num = random.randint(10, 30)
        h, w = img.shape[:2]
        for i in range(cut_num):
            size = random.randint(10, 20)
            x, y = random.randint(0, w-20), random.randint(0, h-20)

            img[y: y+size, x:x+size] = [random.randint(0, 255) for _ in range(3)]

        return Image.fromarray(img)



class RandomTranslation(object):
    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if random.random() < 0.5:
            img2, label2 = self.translation(img2, label2)
        return img1, img2, label1, label2

    def translation(self, img, label):
        img = np.array(img)
        label = np.array(label)
        w_translation = random.randint(-8, 8)
        h_translation = random.randint(-8, 8)
        rows, cols, channels = img.shape
        M = np.float32([[1, 0, w_translation], [0, 1, h_translation]])
        img = cv2.warpAffine(img, M, (cols, rows))
        label = cv2.warpAffine(label, M, (cols, rows))

        return Image.fromarray(img), Image.fromarray(label)


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if random.random() < 0.3:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            label1 = label1.transpose(Image.FLIP_TOP_BOTTOM)
            label2 = label2.transpose(Image.FLIP_TOP_BOTTOM)
        return img1, img2, label1, label2


class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if random.random() < 0.3:
            rotate_degree = random.choice(self.degree)
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)
            label1 = label1.transpose(rotate_degree)
            label2 = label2.transpose(rotate_degree)

        return img1, img2, label1, label2


class RandomRotate(object):
    def __init__(self, degree=15):
        self.degree = degree

    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if random.random() < 0.3:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img1 = img1.rotate(rotate_degree, Image.BILINEAR)
            img2 = img2.rotate(rotate_degree, Image.BILINEAR)
            label1 = label1.rotate(rotate_degree, Image.NEAREST)
            label2 = label2.rotate(rotate_degree, Image.NEAREST)

        return img1, img2, label1, label2


class RandomScaleCrop(object):
    # def __init__(self, base_size=1024, crop_size=1024, fill=0):
    def __init__(self, base_size=512, crop_size=512, fill=0):   

        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        w, h = img1.size
        self.base_size = w
        self.crop_size = w
        
        if random.random() < 1:
            # short_size = random.randint(int(self.base_size * 0.8), int(self.base_size * 1.2))   
            short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 1.5))   
            # short_size = random.randint(int(self.base_size * 0.3), int(self.base_size * 2))   
            # short_size = random.randint(int(self.base_size * 0.1), int(self.base_size * 1.5))   
            w, h = img1.size
            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            img1 = img1.resize((ow, oh), Image.BILINEAR)
            img2 = img2.resize((ow, oh), Image.BILINEAR)
            label1 = label1.resize((ow, oh), Image.NEAREST)  
            label2 = label2.resize((ow, oh), Image.NEAREST)
            # pad crop
            if short_size < self.crop_size:
                padh = self.crop_size - oh if oh < self.crop_size else 0
                padw = self.crop_size - ow if ow < self.crop_size else 0
                img1 = ImageOps.expand(img1, border=(0, 0, padw, padh), fill=0)   
                img2 = ImageOps.expand(img2, border=(0, 0, padw, padh), fill=0)
                label1 = ImageOps.expand(label1, border=(0, 0, padw, padh), fill=self.fill)
                label2 = ImageOps.expand(label2, border=(0, 0, padw, padh), fill=self.fill)
            # random crop crop_size
            w, h = img1.size
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
            img1 = img1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            img2 = img2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            label1 = label1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            label2 = label2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img1, img2, label1, label2


class RandomExchangeOrder(object):
    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        if random.random() < 0.3:
            return img2, img1, label2, label1

        return img1, img2, label1, label2




class HsvAug(object):
    @staticmethod
    def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4): 
        img = np.array(img)
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        # hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))  
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        # cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)  # no return needed 
        img = Image.fromarray(img)
        return img

    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        img1 = self.augment_hsv(img1)
        img2 = self.augment_hsv(img2)

        return img1, img2, label1, label2


class Blur(object):
    @staticmethod
    def gauss_blur(img):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def resize_and_resize(self, img):
        w, h = img.shape[:2]
        rate = random.uniform(0.4, 0.7)

        img = cv2.resize(img, (int(w*rate), int(h*rate)), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = self.gauss_blur(img)

        return img

    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        random_num = random.random()
        if random_num < 0.3:
            img1 = np.array(img1)
            img2 = np.array(img2)
            img1 = self.resize_and_resize(img1)
            img2 = self.resize_and_resize(img2)
            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)

        return img1, img2, label1, label2


class Hist_aug(object): 
    @staticmethod
    def hist_equalize(img, clahe=True, bgr=False):    
        # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
        if clahe:
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = c.apply(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB

    @staticmethod
    def dist_match(img, hist_refs):
        out = np.zeros_like(img)
        _, _, colorChannel = img.shape
        for i in range(colorChannel):
            hist_ref = hist_refs[..., i]
            hist_img, _ = np.histogram(img[:, :, i], 256)  # get the histogram
            # hist_ref, _ = np.histogram(ref[:, :, i], 256)
            cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
            cdf_ref = np.cumsum(hist_ref)

            for j in range(256):
                tmp = abs(cdf_img[j] - cdf_ref)
                tmp = tmp.tolist()
                idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
                out[:, :, i][img[:, :, i] == j] = idx
        return out

    def __call__(self, sample):
        img1, img2, label1, label2 = sample
        random_num = random.random()
        if random_num < 0.2:  
            img1 = np.array(img1)
            img2 = np.array(img2)
            img1 = self.hist_equalize(img1)
            img2 = self.hist_equalize(img2)
            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)
        elif 0.2 < random_num < 0.3:   
            img1 = np.array(img1)
            img2 = np.array(img2)
            hist_refs1 = np.load("/zq2/software/pr/ChangeDetection/utils/hist_files/test_img1.npy")
            hist_refs2 = np.load("/zq2/software/pr/ChangeDetection/utils/hist_files/test_img2.npy")
            img1 = self.dist_match(img1, hist_refs1)
            img2 = self.dist_match(img2, hist_refs2)
            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)

        return img1, img2, label1, label2


class Normalize(object):
    """
    将 PIL.Image 转换为 numpy 数组，进行数据类型转换和归一化。
    - 输入图像 (Image): 转换并归一化到 [0, 1] 范围。
    - 输入标签 (Label): 转换为 0/1 的二值图。
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # 分支 1: 处理 (img1, img2, label1, label2) 的元组
        if len(sample) == 4:
            img1, img2, label1, label2 = sample

            # --- 图像处理 ---
            # 转为 numpy, 交换维度 (H, W, C) -> (C, H, W), 并归一化到 [0, 1]
            img1_np = np.array(img1, dtype=np.float32).transpose((2, 0, 1)) / 255.0
            img2_np = np.array(img2, dtype=np.float32).transpose((2, 0, 1)) / 255.0

            # --- 标签处理 ---
            # 转为 numpy
            label1_np = np.array(label1, dtype=np.float32)
            label2_np = np.array(label2, dtype=np.float32)

            # 如果标签是3通道的（例如来自某些特定数据集），取其中一个通道
            if len(label1_np.shape) > 2:
                label1_np = label1_np[:, :, 0]  # 通常取第一个通道即可
            if len(label2_np.shape) > 2:
                label2_np = label2_np[:, :, 0]

            # 将 0/255 的标签二值化为 0/1
            # 使用阈值处理比直接除法更安全，可以避免浮点数精度问题
            label1_np[label1_np > 0] = 1
            label2_np[label2_np > 0] = 1

            # --- 断言检查 ---
            _, h, w = img1_np.shape
            assert label1_np.shape == (h, w), f"Label1 shape mismatch: expected ({h},{w}), got {label1_np.shape}"
            assert np.all(np.isin(np.unique(label1_np), [0, 1])), "Label1 must be strictly 0 or 1."

            return img1_np, img2_np, label1_np, label2_np

        # 分支 2: 处理 (img1, img2) 的元组 (通常用于推理)
        elif len(sample) == 2:
            img1, img2 = sample
            img1_np = np.array(img1, dtype=np.float32).transpose((2, 0, 1)) / 255.0
            img2_np = np.array(img2, dtype=np.float32).transpose((2, 0, 1)) / 255.0
            return img1_np, img2_np

        # 如果 sample 格式不对，抛出错误
        else:
            raise ValueError(f"Unsupported sample length: {len(sample)}")


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample) == 4:  
            img1, img2, label1, label2 = sample
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()
            label1 = torch.from_numpy(label1).float()
            label2 = torch.from_numpy(label2).float()
            return img1, img2, label1, label2
        elif len(sample) == 2: 
            img1, img2 = sample
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()

            return img1, img2


class ResizeAndPad(object):
    def __call__(self, sample):
        img1, img2 = sample
        h, w = img1.size
        self.base_size = w
        self.crop_size = w

        import utils.GlobalManager as gm
        oh = ow = gm.get_value("size")

        img1 = img1.resize((ow, oh), Image.BILINEAR)
        img2 = img2.resize((ow, oh), Image.BILINEAR)

        padh = self.crop_size - oh if oh < self.crop_size else 0
        padw = self.crop_size - ow if ow < self.crop_size else 0
        img1 = ImageOps.expand(img1, border=(0, 0, padw, padh), fill=0)  
        img2 = ImageOps.expand(img2, border=(0, 0, padw, padh), fill=0)

        return img1, img2


class HistMatch(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.val_avg_img1_hist = self.get_all_image_hist("/nas_data/litianhan/LEVIR_cut/test/Image1")
        self.val_avg_img2_hist = self.get_all_image_hist("/nas_data/litianhan/LEVIR_cut/test/Image2")

    def __call__(self, sample):
        if len(sample) == 2:  
            img1, img2 = sample
            img1 = np.array(img1)
            img2 = np.array(img2)
            # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

            img1_hist = self.get_img_hist(img1)
            img2_hist = self.get_img_hist(img2)

            bhat1 = cv2.compareHist(img1_hist, self.val_avg_img1_hist, method=cv2.HISTCMP_BHATTACHARYYA)
            bhat2 = cv2.compareHist(img2_hist, self.val_avg_img1_hist, method=cv2.HISTCMP_BHATTACHARYYA)
            if bhat1 > 0.65:
                img1 = self.dist_match(img1, self.val_avg_img1_hist)
            if bhat2 > 0.72:
                img2 = self.dist_match(img2, self.val_avg_img2_hist)

            # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)

            return img1, img2

    @staticmethod
    def get_all_image_hist(img_dir_or_file):  
        all_hist = [None, None, None]

        if os.path.isfile(img_dir_or_file):
            img_file = img_dir_or_file
            img = cv2.imread(img_file)
            for i in range(3):
                hist = cv2.calcHist([img], [i], None, [2 ** 8], [0, 2 ** 8])  
                if all_hist[i] is None:
                    all_hist[i] = hist
                else:
                    all_hist[i] += hist
            img_num = 1
        else:
            img_dir = img_dir_or_file
            names = os.listdir(img_dir)
            for m, name in tqdm(enumerate(names)):
                img = cv2.imread(os.path.join(img_dir, name))
                for i in range(3):
                    hist = cv2.calcHist([img], [i], None, [2 ** 8], [0, 2 ** 8]) 
                    if all_hist[i] is None:
                        all_hist[i] = hist
                    else:
                        all_hist[i] += hist
            img_num = len(os.listdir(img_dir))
        all_image_hist = np.concatenate([all_hist[0], all_hist[1], all_hist[2]], axis=1)
        all_image_hist /= img_num

        return all_image_hist

    @staticmethod
    def get_img_hist(img):
        all_hist = [None, None, None]
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [2 ** 8], [0, 2 ** 8])  
            if all_hist[i] is None:
                all_hist[i] = hist
            else:
                all_hist[i] += hist
        image_hist = np.concatenate([all_hist[0], all_hist[1], all_hist[2]], axis=1)
        return image_hist

    @staticmethod
    def dist_match(img, hist_refs):
        out = np.zeros_like(img)
        _, _, colorChannel = img.shape
        for i in range(colorChannel):
            hist_ref = hist_refs[..., i]
            hist_img, _ = np.histogram(img[:, :, i], 256)  # get the histogram
            # hist_ref, _ = np.histogram(ref[:, :, i], 256)
            cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
            cdf_ref = np.cumsum(hist_ref)

            for j in range(256):
                tmp = abs(cdf_img[j] - cdf_ref)
                tmp = tmp.tolist()
                idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
                out[:, :, i][img[:, :, i] == j] = idx
        return out




with_augment_transforms = transforms.Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomFixRotate(),
    RandomRotate(),
    RandomScaleCrop(),
    RandomExchangeOrder(),
    Normalize(),
    ToTensor(),
])

without_augment_transforms = transforms.Compose([
    Normalize(),
    ToTensor()])


if __name__ == "__main__":
    pass
