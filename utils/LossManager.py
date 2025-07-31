from kornia.metrics import SSIM

from src.loss import PSNRLoss, ms_ssim, edges_loss, compute_projection_loss, PyramidPoolingLoss,focal_mse_loss,_ssim,SSIMLoss

import torch

# 定义全局函数，用于直接返回函数型损失
def get_ms_ssim():
    return ms_ssim


def get_edges_loss():
    return edges_loss


def get_compute_projection_loss():
    return compute_projection_loss


def get_focal_mse_loss():
    return focal_mse_loss


def get_SSIM():
    return _ssim

class LossManager:
    def __init__(self, loss_config):
        """
        初始化损失管理器
        :param loss_config: 配置字典，定义损失函数及其权重
        """
        self.loss_functions = {}
        self.loss_weights = {}

        # 动态加载损失函数
        for loss_name, config in loss_config.items():
            loss_fn = self._get_loss_function(loss_name)
            if loss_fn is not None:
                self.loss_functions[loss_name] = loss_fn
                self.loss_weights[loss_name] = config.get("weight", 1.0)
                print(f"注册损失函数: {loss_name}，权重: {self.loss_weights[loss_name]}")


    def _get_loss_function(self, loss_name):
        """
        根据损失函数名称动态加载
        """
        loss_map = {
            "psnr_loss": PSNRLoss,
            "ssim_loss": get_ms_ssim,  # 返回 ms_ssim 函数
            "edges_loss": get_edges_loss,  # 返回 edges_loss 函数
            "multi_scale_mse": PyramidPoolingLoss,
            "mip_loss": get_compute_projection_loss,  # 返回 compute_projection_loss 函数
            "mse_loss": torch.nn.MSELoss,
            "focal_loss": get_focal_mse_loss,
            "SSIM_loss": SSIMLoss
        }
        loss_fn = loss_map.get(loss_name, None)
        return loss_fn() if callable(loss_fn) else None

    def compute_total_loss(self, outputs, labels):
        """
        计算总损失
        """
        total_loss = 0
        individual_losses = {}

        for loss_name, loss_fn in self.loss_functions.items():
            weight = self.loss_weights.get(loss_name, 1.0)

            # 针对函数型损失（如 ms_ssim、edges_loss、compute_projection_loss）的特殊处理
            if loss_name in ["ssim_loss", "edges_loss", "mip_loss","mse_loss","focal_loss"]:
                    loss_single = loss_fn(outputs, labels)
                    loss_value = loss_single * weight

            else:
                loss_single = loss_fn(outputs, labels)
                loss_value = loss_single * weight

            total_loss += loss_value
            individual_losses[loss_name] = loss_single

        return total_loss, individual_losses
