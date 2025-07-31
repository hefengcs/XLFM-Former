import os
import shutil
import time
import signal
import sys
import psutil
import numpy as np
import yaml




import torchvision.transforms as transforms
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
def cleanup_and_exit(signum, frame):
    print("\n[INFO] Caught CTRL+C (SIGINT). Cleaning up...")
    # 杀死所有子进程
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        print(f"Killing child process PID={child.pid}")
        child.kill()
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, cleanup_and_exit)   # CTRL+C
signal.signal(signal.SIGTERM, cleanup_and_exit)  # kill PID

def main(config_path, inference=False):
    # 加载配置
    config = load_config(config_path)

    # 设置 CUDA
    # 假设 config["training"]["devices"] 是 [0,1,2,3]
    devices = config["training"]["devices"]

    # 将列表转换为字符串
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    #
    # # 检查结果
    # print(f"设置的 CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    import torch
    from src.data_aug import Compose
    from utils.TiffDataModule import TiffDataModule, TiffData_Mask_Module
    from pytorch_lightning.callbacks import EarlyStopping
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    # 检查本地可见 GPU 数量
    print("PyTorch 本地可见 GPU 数量: ", torch.cuda.device_count())
    seed = 42
    pl.seed_everything(seed)
    from utils.model_utils import load_model
    from utils.UNetLightningModule import UNetLightningModule
    from utils.SaveInferenceCallback import SaveInferenceCallback
    def parse_transform(transform_config):
        transform_list = []
        for transform in transform_config:
            for name, params in transform.items():
                if hasattr(globals(), name):
                    transform_list.append(getattr(globals(), name)(**params))
        return Compose(transform_list)
    # 数据变换
    train_transform = parse_transform(config["data_transforms"]["train_transform"])
    val_transform = parse_transform(config["data_transforms"]["val_transform"])

    # 数据集
    label = config.get("experiment", {}).get("label", "default_label")
    main_dir = '/home/LifeSci/wenlab/hefengcs/VCD_torch_gnode05'
    ckpt_dir = os.path.join(main_dir, 'ckpt', label)
    sample_dir = os.path.join(main_dir, 'sample', label)
    log_dir = os.path.join(main_dir, 'logs', label)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    # 复制配置文件到日志目录
    timestamp = time.strftime("%d-%H%M%S")

    # 创建动态命名的文件路径
    config_backup_path = os.path.join(
        log_dir, f"{timestamp}_config.yaml"
    )
    # config_backup_path = os.path.join(log_dir, "config_backup.yaml")
    shutil.copy(config_path, config_backup_path)
    print(f"Configuration file copied to {config_backup_path}")

    # data_module = TiffDataModule(
    #     train_list=config["paths"]["train_list"],
    #     val_list=config["paths"]["val_list"],
    #     batch_size=config["training"]["batch_size"],
    #     train_transform=train_transform,
    #     val_transform=val_transform
    # )

    # val_input_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #
    #     # MinMaxNormalize(val_input_min, val_input_max)
    # ])
    #
    # val_gt_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # MinMaxNormalize(val_gt_min, val_gt_max)
    # ])
    Mask_flag = config["experiment"].get("mask", False)
    if Mask_flag:
        config_ratio_pattern =config["experiment"].get("mask_pattern")
        config_ratio = config["experiment"].get("mask_ratio")
        data_module = TiffData_Mask_Module(
            # train_input_dir=train_input_dir,
            # train_gt_dir=train_gt_dir,
            # val_input_dir=validation_input_dir,
            # val_gt_dir=validation_gt_dir,
            train_list=config["paths"]["train_list"],
            val_list=config["paths"]["val_list"],
            batch_size=config["training"]["batch_size"],
            # train_input_transform=val_transform,
            # train_gt_transform=val_transform,
            train_transform=train_transform,
            val_transform=val_transform,
            ratio_pattern=config_ratio_pattern,
            ratio=config_ratio,
        )
    else:
        view_ratio = config["experiment"].get("view_ratio", 0)
        print("view ratio:")
        print(view_ratio)
        data_module = TiffDataModule(
            # train_input_dir=train_input_dir,
            # train_gt_dir=train_gt_dir,
            # val_input_dir=validation_input_dir,
            # val_gt_dir=validation_gt_dir,
            train_list=config["paths"]["train_list"],
            val_list=config["paths"]["val_list"],
            batch_size=config["training"]["batch_size"],
            # train_input_transform=val_transform,
            # train_gt_transform=val_transform,
            train_transform=train_transform,
            val_transform=val_transform,
            view_ratio=view_ratio,
        )

    # model_instance = load_model(config["experiment"]["model"],pretrained=True)

    model_instance = load_model(config["experiment"]["model"])
    print("learning_rate:")
    print(float(config["training"]["learning_rate"]))
    # 初始化 Lightning 模块
    if "PSF_flag" in config["experiment"]["loss_config"]:
        PSF_ratio = np.float32(config["experiment"]["loss_config"]["PSF_ratio"])
        PSF_robust_path = config.get("experiment", {}).get("loss_config", {}).get("PSF_robust_path", None)
        print(PSF_ratio)
        print("PSF_robust_path:",PSF_robust_path)
        model = UNetLightningModule(
            model=model_instance,
            loss_config=config["experiment"]["loss_config"],
            learning_rate=float(config["training"]["learning_rate"]),
            PSF_flag=True,
            batch_size=config["training"]["batch_size"],
            PSF_ratio=PSF_ratio,
            PSF_robust_path=PSF_robust_path,
        )
    else:
        model = UNetLightningModule(
            model=model_instance,
            loss_config=config["experiment"]["loss_config"],
            learning_rate=float(config["training"]["learning_rate"]),
            batch_size=config["training"]["batch_size"],
        )

    if "pretrained" in config["experiment"]:
        # model_instance = load_model(
        #     config["experiment"]["model"],
        #     pretrained=config["experiment"].get("pretrained", "False").lower() == "true"
        # )
        checkpoint_path = config["experiment"]["pretrained"]
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model_state_dict = model.state_dict()
        checkpoint_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                                 k in model_state_dict and model_state_dict[k].shape == v.shape}
        model_state_dict.update(checkpoint_state_dict)
        model.load_state_dict(model_state_dict)


    if inference:
        print("进入推理模式...")
        checkpoint_path = config["paths"]["infer_checkpoint_path"]
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model_state_dict = model.state_dict()
        checkpoint_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                                 k in model_state_dict and model_state_dict[k].shape == v.shape}
        model_state_dict.update(checkpoint_state_dict)
        model.load_state_dict(model_state_dict)
    else:
        print("进入训练模式...")

    SIMMIM_path = config["paths"].get("SimMIM_path", None)

    if SIMMIM_path:

        checkpoint_path = SIMMIM_path
        ID =devices[0]
        cuda_ID = "cuda:" +str(ID)
        checkpoint = torch.load(checkpoint_path, map_location=cuda_ID)

        # 获取当前模型的 state_dict
        model_state_dict = model.state_dict()

        # 过滤掉 checkpoint 中的 "head" 部分，同时确保其他权重形状匹配
        checkpoint_state_dict = {k: v for k, v in checkpoint["state_dict"].items()
                                 if k in model_state_dict
                                 and model_state_dict[k].shape == v.shape
                                 and not k.startswith('model.mask_head')}

        # 更新模型的 state_dict
        model_state_dict.update(checkpoint_state_dict)

        # 加载更新后的 state_dict
        model.load_state_dict(model_state_dict)
        print("SIMMIM_path:", checkpoint_path)
    else:
        print("No SIMMIM_path found in config file")


    ImageNet_path = config["paths"].get("ImageNet_path", None)
    if ImageNet_path:
        checkpoint = torch.load(ImageNet_path, map_location="cuda", weights_only=True)
        checkpoint = checkpoint["model"]  # 进入 "model" 键，获取实际的 state_dict

        model_state_dict = model.state_dict()  # 获取当前模型的参数字典

        # 修正 checkpoint 的键名前缀，使其匹配你的模型
        checkpoint = {f"model.restoration_net.{k}": v for k, v in checkpoint.items()}

        # 过滤掉不匹配的层
        checkpoint_state_dict = {k: v for k, v in checkpoint.items()
                                 if k in model_state_dict and model_state_dict[k].shape == v.shape}

        print(f"Successfully loaded {len(checkpoint_state_dict)} layers from {ImageNet_path}")
        print("Loaded layers:", list(checkpoint_state_dict.keys()))

        # 加载权重
        model_state_dict.update(checkpoint_state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        print(f"Loaded ImageNet weights from {ImageNet_path}")

    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证集损失
        patience=10,  # 允许 5 个 epoch 性能不改善
        mode='min',  # 希望监控指标越小越好
        # min_delta=1e-8,  # 改善的最小幅度
        verbose=True
    )

    if inference:
        save_flag = config["callbacks"]["need_save"]
        if save_flag:
            save_inference_callback = SaveInferenceCallback(sample_dir=sample_dir, is_inference=True)
    else:
        save_inference_callback = SaveInferenceCallback(sample_dir=sample_dir,
                                                        epoch_interval=config["callbacks"]["epoch_interval"])
    # 回调函数
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=50, #每500 step就保存一次
        dirpath=ckpt_dir,
        save_top_k=config["callbacks"]["save_top_k"],
        monitor=config["callbacks"]["monitor"],
        mode=config["callbacks"]["mode"],
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(log_dir)

    # 是否恢复训练
    #resume_path = config["paths"].get("Resume_path", None)
    # if resume_path:
    #     print("Training from resume_path:",resume_path)
    #     trainer = pl.Trainer(
    #         max_epochs=config["training"]["max_epochs"],
    #         devices=config["training"]["devices"],
    #         accelerator=config["training"]["accelerator"],
    #         strategy=config["training"]["strategy"],
    #         # callbacks=[checkpoint_callback, lr_monitor,save_inference_callback, early_stopping],
    #         callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    #         logger=logger,
    #         resume_from_checkpoint=resume_path
    #     )
    #
    # else:
    # # Trainer
    #     print("Training from scratch...")
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        devices=config["training"]["devices"],
        #devices=[0,1,2,3],
        accelerator=config["training"]["accelerator"],
        strategy=config["training"]["strategy"],
        # callbacks=[checkpoint_callback, lr_monitor,save_inference_callback, early_stopping],
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=logger,

    )

    if inference:
        trainer.validate(model, datamodule=data_module)
    else:
        # 是否恢复训练
        resume_path = config["paths"].get("Resume_path", None)
        if resume_path:
            print("Training from resume_path:", resume_path)
            trainer.fit(model, datamodule=data_module,ckpt_path=resume_path)
        else:
            print("Training from scratch...")
            trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, default="./configs/infer.yaml", required=True, help="Path to config file")
    parser.add_argument("--inference", action="store_true", default="", help="Run inference instead of training")
    args = parser.parse_args()
    # main(args.config)
    main(config_path=args.config, inference=args.inference)