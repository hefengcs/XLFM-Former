import os
import yaml
from src.data_aug import Compose
from utils.TiffDataModule import TiffDataModule
import torchvision.transforms as transforms
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_transform(transform_config):
    transform_list = []
    for transform in transform_config:
        for name, params in transform.items():
            if hasattr(globals(), name):
                transform_list.append(getattr(globals(), name)(**params))
    return Compose(transform_list)


def main(config_path, inference=False):
    # 加载配置
    config = load_config(config_path)

    # 设置 CUDA
    #os.environ["CUDA_VISIBLE_DEVICES"] = config["training"]["gpu_ids"]
    #print(f"设置的 CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    seed = 42
    pl.seed_everything(seed)
    from utils.model_utils import load_model
    from utils.UNetLightningModule import UNetLightningModule
    from utils.SaveInferenceCallback import SaveInferenceCallback
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

    # data_module = TiffDataModule(
    #     train_list=config["paths"]["train_list"],
    #     val_list=config["paths"]["val_list"],
    #     batch_size=config["training"]["batch_size"],
    #     train_transform=train_transform,
    #     val_transform=val_transform
    # )

    val_input_transform = transforms.Compose([
        transforms.ToTensor(),

        # MinMaxNormalize(val_input_min, val_input_max)
    ])

    val_gt_transform = transforms.Compose([
        transforms.ToTensor(),
        # MinMaxNormalize(val_gt_min, val_gt_max)
    ])

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

    )

    # 动态加载模型
    model_instance = load_model(config["experiment"]["model"])

    # 初始化 Lightning 模块
    model = UNetLightningModule(
        batch_size=config["training"]["batch_size"],
        model=model_instance,
        loss_config=config["experiment"]["loss_config"],
        learning_rate=float(config["training"]["learning_rate"], )
    )

    if inference:
        print("进入推理模式...")
        checkpoint_path = config["paths"]["infer_checkpoint_path"]
        checkpoint = torch.load(checkpoint_path)
        model_state_dict = model.state_dict()
        checkpoint_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                                 k in model_state_dict and model_state_dict[k].shape == v.shape}
        model_state_dict.update(checkpoint_state_dict)
        model.load_state_dict(model_state_dict)
    else:
        print("进入训练模式...")

    save_flag = config["callbacks"]["need_save"]

    save_inference_callback = SaveInferenceCallback(sample_dir=sample_dir, is_inference=save_flag)

    # 回调函数
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=config["callbacks"]["save_top_k"],
        monitor=config["callbacks"]["monitor"],
        mode=config["callbacks"]["mode"],

    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(log_dir)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        devices=config["training"]["devices"],
        accelerator=config["training"]["accelerator"],
        strategy=config["training"]["strategy"],
        callbacks=[checkpoint_callback, lr_monitor, save_inference_callback],
        # callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
    )

    if inference:
        trainer.validate(model, datamodule=data_module)
        #trainer.predict(model, datamodule=data_module)
    else:
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, default="configs/infer/infer_small_raw.yaml", required=True,
                        help="Path to config file")
    parser.add_argument("--inference", action="store_true", default="True", help="Run inference instead of training")
    args = parser.parse_args()
    # main(args.config)
    main(config_path=args.config, inference=args.inference)

