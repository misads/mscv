# encoding=utf-8
"""
    重叠取块TTA。
    用法：
    >>> for i, data in enumerate(test_dataloader):
    >>>     img, paths = data['input'], data['path']
    >>>     img = img.to(device=opt.device)
    >>>     model = get_model()
    >>>     output = tta_inference(model, img, 10, 10, 256, 256)

"""

import pdb

import torch
from torch.autograd import Variable
from torchvision.transforms import transforms, ToPILImage, ToTensor
from .image import tensor2im
from PIL import Image
from misc_utils import progress_bar


class OverlapTTA(object):
    """overlap TTA

    Args:
        nw(int): num of patches (in width direction)
        nh(int):  num of patches (in height direction)
        patch_w(int): width of a patch.
        patch_h(int): height of a patch.
        padding_w(int): 膨胀预测，padding的部分会被预测但不会计算到结果中，只会保留中心的部分，padding后每张小图的宽度会增加 2 * padding_w.
        padding_h(int): Both side of height will be padded with padding_h.

    """
    def __init__(self, img, nw, nh, patch_w=256, patch_h=256, padding_w=0, padding_h=0, transforms=None):

        self.img = img
        self.nw = nw
        self.nh = nh
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.N, self.C, self.H, self.W = img.shape
        self.transforms = transforms

        self.device = img.device

        #####################################
        #                 步长
        #####################################
        stride_h = (self.H - patch_h) // (nh - 1)
        stride_w = (self.W - patch_w) // (nw - 1)
        self.padding_w = padding_w
        self.padding_h = padding_h

        self.overlap_times = torch.zeros((self.C, self.H, self.W)).cpu()
        self.slice_h = []
        self.slice_w = []

        #####################################
        #   除了最后一个patch, 都按照固定步长取块
        # 将位置信息先保存在slice_h和slice_w数组中
        #####################################
        for i in range(nh - 1):
            self.slice_h.append([i * stride_h + padding_h, i * stride_h + patch_h + padding_h])
        self.slice_h.append([self.H - patch_h, self.H])
        for i in range(nw - 1):
            self.slice_w.append([i * stride_w + padding_w, i * stride_w + patch_w + padding_w])
        self.slice_w.append([self.W - patch_w, self.W])

        #####################################
        #             保存结果的数组
        #####################################
        if self.padding_w != 0 or self.padding_h != 0:
            self.result = torch.zeros((self.C, self.H + padding_h * 2, self.W + padding_w * 2)).cpu()
        else:
            self.result = torch.zeros((self.C, self.H, self.W)).cpu()

    def collect(self, x, cur):
        x = x.detach().cpu()

        j = cur % self.nw
        i = (cur - j) // self.nw

        #####################################
        #         分别记录图像和重复次数
        #####################################

        self.result[:, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]] += x
        self.overlap_times[:, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]] += 1

    def combine(self):
        return (self.result / self.overlap_times)[:, self.padding_h:self.H + self.padding_h,
                                                  self.padding_w:self.W + self.padding_w]

    def __getitem__(self, index):
        """
            获取tta patch作为网络输入
            :param index:
            :return:
        """
        j = index % self.nw
        i = index // self.nw
        if self.padding_w != 0 or self.padding_h != 0:
            img = self.img[:, :, self.slice_h[i][0] - self.padding_h:self.slice_h[i][1] + self.padding_h,
                           self.slice_w[j][0] - self.padding_w:self.slice_w[j][1] + self.padding_w]
        else:
            img = self.img[:, :, self.slice_h[i][0]:self.slice_h[i][1], self.slice_w[j][0]:self.slice_w[j][1]]

        if self.transforms is not None:
            img = self.transforms(img[0]).unsqueeze(dim=0)

        img_var = Variable(img, requires_grad=False).to(device=self.device)
        return img_var

    def __len__(self):
        return self.nw * self.nh


def tta_inference(forward_func, img, nw, nh, patch_w=256, patch_h=256,
                  padding_w=0, padding_h=0, progress_idx=None):
    tta = OverlapTTA(img, nw, nh, patch_w, patch_h, padding_w, padding_h)

    with torch.no_grad():
        for j, x in enumerate(tta):  # 获取每个patch输入
            if progress_idx is not None:
                idx, tot = progress_idx
                progress_bar(idx * len(tta) + j, tot * len(tta), 'TTA... ')
            generated = forward_func(x)
            # torch.cuda.empty_cache()
            tta.collect(generated[0], j)  # 收集inference结果
        output = tta.combine()

    return output


def tta_inference_x8(forward_func, img, nw, nh, patch_w=256, patch_h=256,
                     padding_w=0, padding_h=0, progress_idx=None):

    assert padding_w == padding_h, 'tta_x8 mode requires padding_w==padding_h'

    N, C, H, W = img.shape
    result = torch.zeros((C, H, W)).cpu()
    device = img.device
    transpose_pairs = {0: 0, 1: 1, 2: 4, 3: 3, 4: 2, 5: 5, 6: 6}

    to_image = ToPILImage()
    to_tensor = ToTensor()

    img_cpu = to_image(img.cpu()[0])
    for i in range(8):
        if i != 7:  # i=7是不变换
            img_transposed = img_cpu.transpose(i)
        else:
            img_transposed = img_cpu

        if progress_idx is not None:
            idx, tot = progress_idx
            new_progress_idx = (idx * 8 + i, tot * 8)
        else:
            new_progress_idx = None

        img_transposed = to_tensor(img_transposed).unsqueeze(0).to(device)

        res_transposed = tta_inference(forward_func, img_transposed, nw, nh, patch_w, patch_h,
                                       padding_w, padding_h, new_progress_idx).unsqueeze(0)
        res_transposed = tensor2im(res_transposed)
        res_transposed = Image.fromarray(res_transposed)
        if i != 7:
            tta = res_transposed.transpose(transpose_pairs[i])  # 转回去
        else:
            tta = res_transposed

        tta = to_tensor(tta)

        result += tta

    return result / 8
