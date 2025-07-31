import torch


def process_image_tensor(tensor_image, lenses_info):
    """
    处理输入的张量图像，按照 lenses_info 提供的透镜区域信息进行裁剪并堆叠。

    Args:
        tensor_image (torch.Tensor): 输入图像的张量 (C, H, W)，例如 (1, 2304, 2304)。
        lenses_info (list of tuples): 每个透镜区域的 (x, y, w, h) 坐标信息。

    Returns:
        torch.Tensor: 堆叠后的裁剪图像 (N, C, 600, 600)，N 是透镜的数量。
    """

    # 假设 tensor_image 为 3D 张量 (C, H, W)
    C, H, W = tensor_image.shape

    cropped_images = []

    for x, y, w, h in lenses_info:
        # 裁剪透镜区域
        lens_image = tensor_image[:, y:y + h, x:x + w]

        # 检查图像尺寸，必要时进行填充
        if lens_image.shape[1] < 600 or lens_image.shape[2] < 600:
            new_image = torch.zeros((C, 600, 600), dtype=tensor_image.dtype, device=tensor_image.device)  # 创建空白图像
            offset_y = (600 - lens_image.shape[1]) // 2
            offset_x = (600 - lens_image.shape[2]) // 2
            new_image[:, offset_y:offset_y + lens_image.shape[1], offset_x:offset_x + lens_image.shape[2]] = lens_image
            cropped_images.append(new_image)
        else:
            # 如果已经是 600x600 以上的尺寸，直接使用
            cropped_images.append(lens_image)

    # 将裁剪后的图像堆叠成一个新的 tensor
    stacked_images = torch.stack(cropped_images, dim=0)  # 堆叠后维度 (N, C, 600, 600)

    return stacked_images

coordinates_path = '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05/data_prepare/coordinates.txt'  # 坐标信息文件路径

# 生成一个随机的 3D 张量作为输入
tensor_image = torch.randn(1,2048, 2048)
lenses_info = []
with open(coordinates_path, 'r') as file:
    for line in file:
        x, y, w, h = map(int, line.split())
        lenses_info.append((x, y, w, h))

output_tensor = process_image_tensor(tensor_image, lenses_info)
print(output_tensor.shape)