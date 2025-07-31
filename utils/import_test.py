try:
    # 尝试从全局的 basicsr 模块中导入 NAFNet_arch 和 NAFNet
    from basicsr.models.archs import NAFNet_arch
    from basicsr.models.archs.NAFNet_arch import NAFNet
    print("NAFNet module imported successfully from global basicsr.")
except ImportError as e:
    print(f"Error importing NAFNet or NAFNet_arch: {e}")
