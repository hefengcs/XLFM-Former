# model_utils.py
import importlib

# 动态加载模型
def load_model(model_name, **kwargs):
    try:
        # 从指定模块加载模型
        module = importlib.import_module("utils.head")  # 替换为模型所在的实际路径
        model_class = getattr(module, model_name, None)
        if model_class is None:
            raise ValueError(f"模型 {model_name} 未在模块中定义！")
        return model_class(**kwargs)  # 支持传递额外参数
    except Exception as e:
        raise ValueError(f"加载模型失败: {model_name}. 错误: {e}")
