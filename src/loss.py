import  os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import tifffile as tiff
import torch
import torch.nn as nn
import pywt

class PSNRLoss(torch.nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, x, y):
        mse = torch.mean((x - y) ** 2)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return -psnr  # Note: We return negative PSNR for minimization

def sobel_edges(input):
    '''
    find the edges of the input image using Sobel filter

    Params:
        -input : tensor of shape [batch, channels, height, width]
    return:
        -tensor of the edges: [batch, channels, height, width, 2] (dx and dy)
    '''
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(input.device)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(input.device)

    # 将卷积核扩展到输入的通道数
    sobel_x = sobel_x.expand(input.size(1), 1, 3, 3)
    sobel_y = sobel_y.expand(input.size(1), 1, 3, 3)

    grad_x = F.conv2d(input, sobel_x, padding=1, groups=input.size(1))
    grad_y = F.conv2d(input, sobel_y, padding=1, groups=input.size(1))

    return torch.stack([grad_x, grad_y], dim=-1)


def l2_loss(image, reference):
    return torch.mean((image - reference) ** 2)


def edges_loss(image, reference):
    '''
    params:
        -image : tensor of shape [batch, channels, height, width], the output of the network
        -reference : same shape as the image
    '''
    edges_sr = sobel_edges(image)
    edges_hr = sobel_edges(reference)
    return l2_loss(edges_sr, edges_hr)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def gaussian_window(self, window_size, sigma):
        gauss = torch.tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window_size=11, size_average=True):
        (_, channel, height, width) = img1.size()
        #( channel, height, width) = img1.size()
        window = self.create_window(window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        return self.ssim(img1, img2, self.window_size, self.size_average)


def consistency(matrix1, PSF_fft, device):

        def pad_and_assign(matrix, target_size):
            # 在 GPU 上创建张量
            padded_matrix = torch.zeros(target_size, dtype=matrix.dtype, device=device)
            padded_matrix[:, 852:1452, 852:1452] = matrix
            return padded_matrix

        def fourier_conv_batch(images, fft_psf):


            fft_images = torch.fft.fft2(images)
            result = torch.real(torch.fft.ifft2(fft_images * fft_psf))
            return result

        target_size = (300, 2304, 2304)
        padded_matrix1 = pad_and_assign(matrix1, target_size)

        summed_matrix = fourier_conv_batch(padded_matrix1, PSF_fft)
        #沿着第0个维度加和sum
        summed_matrix = summed_matrix.sum(dim=0)
        cropped_matrix = summed_matrix[128:-128, 128:-128]
        cropped_matrix = cropped_matrix.unsqueeze(0).unsqueeze(3)
        cropped_matrix = cropped_matrix / cropped_matrix.max()

        return cropped_matrix

def consistency_loss(input, target, PSF):

        criterion = nn.MSELoss()
        device = PSF.device

        with torch.no_grad():
            # input = input.to(device)
            # target = target.to(device)

            infer_watch = consistency(input, PSF, device)
            gt_watch = consistency(target, PSF, device)

        value = criterion(infer_watch, gt_watch)

        return value


def consistency_loss_log(input, target, PSF, logger, global_step):
    criterion = nn.MSELoss()
    device = PSF.device
    writer = logger.experiment

    #with torch.no_grad():
        # input = input.to(device)
        # target = target.to(device)

    infer_watch = consistency(input, PSF, device)
    gt_watch = consistency(target, PSF, device)

    value = criterion(infer_watch, gt_watch)

    # 记录到 TensorBoard
    if global_step % 200 == 0:
        writer.add_image('Input/infer_watch', infer_watch[:,:,:,0], global_step)
        writer.add_image('Target/gt_watch', gt_watch[:,:,:,0], global_step)

    return value


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()



#带mask
# class PSNRLoss(nn.Module):
#     def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
#         super(PSNRLoss, self).__init__()
#         assert reduction == 'mean'
#         self.loss_weight = loss_weight
#         self.scale = 10 / np.log(10)
#         self.toY = toY
#         self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
#         self.first = True
#
#     def forward(self, pred, target):
#         assert pred.dim() == 4 and target.dim() == 4
#
#         if self.toY:
#             if self.first:
#                 self.coef = self.coef.to(pred.device)
#                 self.first = False
#             pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
#             target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
#             pred, target = pred / 255., target / 255.
#
#         # 仅使用 target > 0 的位置计算
#         mask = (target > 0).float()
#         num_valid = mask.sum(dim=(1, 2, 3)) + 1e-8  # 防止除0
#
#         mse = ((pred - target) ** 2 * mask).sum(dim=(1, 2, 3)) / num_valid
#         psnr = self.scale * torch.log(mse + 1e-8)
#
#         return self.loss_weight * psnr.mean()








import torchvision.models as models
class PerceptualLoss(nn.Module):
    def __init__(self, pretrained_model, layers):
        super(PerceptualLoss, self).__init__()
        self.model = pretrained_model
        self.layers = layers
        self.features = []
        self.hooks = []

        for idx, layer in enumerate(self.model):
            if idx in self.layers:
                hook = layer.register_forward_hook(self.save_feature)
                self.hooks.append(hook)

    def save_feature(self, module, input, output):
        self.features.append(output)

    def forward(self, input, target):
        self.features = []
        self.model(input)
        input_features = self.features

        self.features = []
        self.model(target)
        target_features = self.features

        loss = 0
        for input_feature, target_feature in zip(input_features, target_features):
            loss += F.l1_loss(input_feature, target_feature)

        return loss



class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        layer_idxs = [3, 8, 17, 26]  # 对应'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'的层索引

        self.perceptual_loss = PerceptualLoss(vgg, layer_idxs)

    def forward(self, input, target):
        # input 和 target 的形状是 (batch_size, channels, height, width)
        batch_size, channels, height, width = input.size()

        total_loss = 0
        for i in range(channels):
            input_channel = input[:, i:i+1, :, :].expand(batch_size, 3, height, width)
            target_channel = target[:, i:i+1, :, :].expand(batch_size, 3, height, width)
            loss = self.perceptual_loss(input_channel, target_channel)
            total_loss += loss

        return total_loss


def max_intensity_projection(tensor, dim):
    """
    计算给定维度上的最大强度投影。

    参数:
        tensor (torch.Tensor): 输入的三维 tensor，形状为 (D, H, W)。
        dim (int): 要投影的维度，0 (Z方向), 1 (Y方向), 2 (X方向)。

    返回:
        torch.Tensor: 投影后的二维 tensor。
    """
    return torch.max(tensor, dim=dim).values


#def compute_projection_loss(prediction, target, loss_fn=nn.MSELoss()):
def compute_projection_loss(prediction, target, loss_fn=nn.MSELoss()):
    """
    计算预测和目标在三个方向上的最大投影损失。

    参数:
        prediction (torch.Tensor): 模型预测的三维 tensor，形状为 (D, H, W)。
        target (torch.Tensor): 目标的三维 tensor，形状为 (D, H, W)。
        loss_fn (nn.Module): 损失函数，例如 nn.MSELoss()。

    返回:
        torch.Tensor: 合并后的总损失。
    """
    # 计算 X、Y、Z 三个方向的投影
    pred_proj_z = max_intensity_projection(prediction, dim=0)  # Z方向投影
    pred_proj_y = max_intensity_projection(prediction, dim=1)  # Y方向投影
    pred_proj_x = max_intensity_projection(prediction, dim=2)  # X方向投影

    target_proj_z = max_intensity_projection(target, dim=0)
    target_proj_y = max_intensity_projection(target, dim=1)
    target_proj_x = max_intensity_projection(target, dim=2)

    # 分别计算每个方向的投影损失
    loss_z = loss_fn(pred_proj_z, target_proj_z)
    loss_y = loss_fn(pred_proj_y, target_proj_y)
    loss_x = loss_fn(pred_proj_x, target_proj_x)

    # 合并损失（可以根据任务需求对不同方向赋予不同的权重）
    total_loss = loss_z + loss_y + loss_x

    return total_loss


import torch
import torch.nn.functional as F
import math


# Gaussian window generation
def gaussian(window_size, sigma):
    """Generate a Gaussian kernel."""
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# Create a 2D Gaussian window
def create_window(window_size, channel):
    """Create a 2D Gaussian window for SSIM calculation."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size)
    return window


# SSIM calculation for one scale
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Helper function to calculate SSIM."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# MS-SSIM calculation
def ms_ssim(img1, img2, window_size=11, size_average=True):
    """Multi-Scale SSIM calculation."""
    device = img1.device
    channel = img1.size(1)
    window = create_window(window_size, channel).to(device)

    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # Predefined weights for MS-SSIM
    levels = len(weights)

    msssim = []
    for _ in range(levels):
        ssim_val = _ssim(img1, img2, window, window_size, channel, size_average)
        msssim.append(ssim_val)

        # Downsampling the images for next scale
        img1 = F.avg_pool2d(img1, kernel_size=2)
        img2 = F.avg_pool2d(img2, kernel_size=2)

    # Combine results from all levels
    msssim_val = torch.stack(msssim)
    msssim_val =1- msssim_val
    return torch.prod(msssim_val ** torch.Tensor(weights).to(device))

def compute_projection_perceptual_loss(prediction, target, loss_fn=VGGPerceptualLoss()):
    """
    计算预测和目标在三个方向上的最大投影损失。

    参数:
        prediction (torch.Tensor): 模型预测的三维 tensor，形状为 (D, H, W)。
        target (torch.Tensor): 目标的三维 tensor，形状为 (D, H, W)。
        loss_fn (nn.Module): 损失函数，例如 nn.MSELoss()。

    返回:
        torch.Tensor: 合并后的总损失。
    """
    # 计算 X、Y、Z 三个方向的投影
    pred_proj_z = max_intensity_projection(prediction, dim=0)  # Z方向投影
    pred_proj_y = max_intensity_projection(prediction, dim=1)  # Y方向投影
    pred_proj_x = max_intensity_projection(prediction, dim=2)  # X方向投影

    target_proj_z = max_intensity_projection(target, dim=0)
    target_proj_y = max_intensity_projection(target, dim=1)
    target_proj_x = max_intensity_projection(target, dim=2)

    # 分别计算每个方向的投影损失
    loss_z = loss_fn(pred_proj_z, target_proj_z)
    loss_y = loss_fn(pred_proj_y, target_proj_y)
    loss_x = loss_fn(pred_proj_x, target_proj_x)

    # 合并损失（可以根据任务需求对不同方向赋予不同的权重）
    total_loss = loss_z + loss_y + loss_x

    return total_loss



import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile as tiff

def haar_wavelet_transform_2d(x):
    """
    对输入张量进行二维 Haar 小波变换。
    输入：
        x: PyTorch 张量，形状为 (batch_size, channels, height, width)
    输出：
        LL, LH, HL, HH: 四个张量，分别为 Haar 小波变换的低频和高频分量
    """
    batch_size, channels, height, width = x.size()

    # Haar 小波滤波器 (2D)
    haar_filter = torch.tensor([[[[1, 1], [1, 1]]], [[[1, -1], [1, -1]]],
                                [[[1, 1], [-1, -1]]], [[[1, -1], [-1, 1]]]], dtype=x.dtype, device=x.device) / 2.0

    # 扩展滤波器维度以匹配输入通道数
    haar_filter = haar_filter.repeat(channels, 1, 1, 1)

    # 准备存储结果的张量
    LL = torch.zeros((batch_size, channels, height // 2, width // 2), dtype=x.dtype, device=x.device)
    LH = torch.zeros((batch_size, channels, height // 2, width // 2), dtype=x.dtype, device=x.device)
    HL = torch.zeros((batch_size, channels, height // 2, width // 2), dtype=x.dtype, device=x.device)
    HH = torch.zeros((batch_size, channels, height // 2, width // 2), dtype=x.dtype, device=x.device)

    # 对每个图像应用2D卷积
    for i in range(channels):
        LL[:, i, :, :] = F.conv2d(x[:, i:i+1, :, :], haar_filter[0].unsqueeze(0), stride=2, padding=0)
        LH[:, i, :, :] = F.conv2d(x[:, i:i+1, :, :], haar_filter[1].unsqueeze(0), stride=2, padding=0)
        HL[:, i, :, :] = F.conv2d(x[:, i:i+1, :, :], haar_filter[2].unsqueeze(0), stride=2, padding=0)
        HH[:, i, :, :] = F.conv2d(x[:, i:i+1, :, :], haar_filter[3].unsqueeze(0), stride=2, padding=0)

    return LL, LH, HL, HH

class HaarMSELoss(nn.Module):
    def __init__(self):
        super(HaarMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        # 对输入和目标分别进行 Haar 小波变换
        input_LL, input_LH, input_HL, input_HH = haar_wavelet_transform_2d(input)
        target_LL, target_LH, target_HL, target_HH = haar_wavelet_transform_2d(target)

        # 计算各个分量的 MSE
        loss_LL = self.mse_loss(input_LL, target_LL)
        loss_LH = self.mse_loss(input_LH, target_LH)
        loss_HL = self.mse_loss(input_HL, target_HL)
        loss_HH = self.mse_loss(input_HH, target_HH)

        # 总损失为各个分量损失的加权和
        total_loss = loss_LL + loss_LH + loss_HL + loss_HH
        return total_loss



def multi_scale_mse(pred, target, scales=[1, 0.5, 0.25], weights=None):
    """
    Compute multi-scale MSE between two tensors.

    Args:
        pred (torch.Tensor): Predicted tensor of shape (N, C, H, W).
        target (torch.Tensor): Ground truth tensor of shape (N, C, H, W).
        scales (list): List of scales to apply for downsampling. Default is [1, 0.5, 0.25].
        weights (list): List of weights for each scale. If None, equal weights are used.

    Returns:
        torch.Tensor: The multi-scale MSE loss.
    """
    # If no weights are provided, use equal weights
    if weights is None:
        weights = [1.0] * len(scales)

    # Make sure the weights are normalized
    weights = [w / sum(weights) for w in weights]

    # Initialize the loss
    total_loss = 0.0

    # Compute MSE for each scale
    for scale, weight in zip(scales, weights):
        if scale != 1:
            # Downsample the tensors to the desired scale
            pred_scaled = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
            target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            # Use the original tensors for scale = 1
            pred_scaled = pred
            target_scaled = target

        # Compute MSE for the current scale
        mse_loss = F.mse_loss(pred_scaled, target_scaled)

        # Weight the MSE loss and add to the total loss
        total_loss += weight * mse_loss

    return total_loss

class PyramidPoolingLoss(torch.nn.Module):
    def __init__(self, scales=[1, 0.5, 0.25, 0.125]):
        super(PyramidPoolingLoss, self).__init__()
        self.scales = scales

    def forward(self, pred, target):
        total_loss = 0
        for scale in self.scales:
            pred_scaled = F.adaptive_avg_pool2d(pred, output_size=(int(pred.size(2) * scale), int(pred.size(3) * scale)))
            target_scaled = F.adaptive_avg_pool2d(target, output_size=(int(target.size(2) * scale), int(target.size(3) * scale)))
            total_loss += F.mse_loss(pred_scaled, target_scaled)
        return total_loss


def grid_mse_loss(recon, target, grid_size=(5, 10, 10), stride=(5, 10, 10)):
    batch_size, d, h, w = recon.size()
    grid_d, grid_h, grid_w = grid_size
    stride_d, stride_h, stride_w = stride

    # 使用 unfold 来提取网格块
    recon_unfold = recon.unfold(1, grid_d, stride_d).unfold(2, grid_h, stride_h).unfold(3, grid_w, stride_w)
    target_unfold = target.unfold(1, grid_d, stride_d).unfold(2, grid_h, stride_h).unfold(3, grid_w, stride_w)

    # 将展开的维度 reshape 以便计算损失
    recon_patches = recon_unfold.contiguous().view(-1, grid_d, grid_h, grid_w)
    target_patches = target_unfold.contiguous().view(-1, grid_d, grid_h, grid_w)

    # 对每个网格块内部进行加和
    recon_patch_sum = recon_patches.sum(dim=(1, 2, 3))  # 每个网格块内部的和
    target_patch_sum = target_patches.sum(dim=(1, 2, 3))  # 每个网格块内部的和

    # 计算每个网格块和之间的 MSE 损失
    loss = nn.functional.mse_loss(recon_patch_sum, target_patch_sum)

    return loss

# 示例用法
#focal_loss
def focal_mse_loss(pred, target, gamma=2.0, reduction='mean'):
    mse = (pred - target) ** 2
    weights = ((target - pred).abs() / target.clamp(min=1e-6)).clamp(max=10)
    focal_weights = (weights ** gamma)
    loss = focal_weights * mse
    return loss.mean() if reduction == 'mean' else loss








if __name__ == "__main__":
    recon = torch.rand(1,  300, 600, 600).cuda()  # (batch_size, channels, depth, height, width)
    target = torch.rand(1,  300, 600, 600).cuda()
    grid_size = (5, 10, 10)
    stride = (2, 5, 5)  # 控制重叠的步幅
    for i in range(10):
        loss = focal_mse_loss(recon, target)
        print(loss)