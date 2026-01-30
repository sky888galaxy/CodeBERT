#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端环境检查与部署工具
检验 Python 环境、依赖、数据是否准备就绪
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Tuple, Dict, List


class EnvironmentChecker:
    """云端环境检查器"""
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
    
    def check_python_version(self) -> bool:
        """检查 Python 版本 >= 3.8"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            self.checks_passed.append(f"✓ Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.checks_failed.append(f"✗ Python 版本过低 ({version.major}.{version.minor}，需 >= 3.8)")
            return False
    
    def check_library(self, lib_name: str, import_name: str = None) -> bool:
        """检查库是否安装"""
        if import_name is None:
            import_name = lib_name
        
        try:
            __import__(import_name)
            # 获取版本信息
            if import_name == 'torch':
                import torch
                version = torch.__version__
                cuda_available = torch.cuda.is_available()
                status = f"✓ {lib_name} {version} (CUDA: {'可用' if cuda_available else '不可用'})"
            elif import_name == 'transformers':
                import transformers
                version = transformers.__version__
                status = f"✓ {lib_name} {version}"
            elif import_name == 'sklearn':
                import sklearn
                version = sklearn.__version__
                status = f"✓ scikit-learn {version}"
            else:
                status = f"✓ {lib_name}"
            
            self.checks_passed.append(status)
            return True
        except ImportError:
            self.checks_failed.append(f"✗ {lib_name} 未安装")
            return False
    
    def check_gpu(self) -> bool:
        """检查 GPU 可用性"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
                total_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                                   for i in range(device_count)) / 1e9
                status = f"✓ GPU 可用: {device_count}× {', '.join(devices)} ({total_memory:.1f} GB)"
                self.checks_passed.append(status)
                return True
            else:
                self.warnings.append("⚠ GPU 不可用，将使用 CPU (训练会很慢)")
                return False
        except Exception as e:
            self.checks_failed.append(f"✗ GPU 检查失败: {e}")
            return False
    
    def check_data_files(self, data_dir: str = "code_module/data") -> bool:
        """检查数据文件是否存在"""
        data_path = Path(data_dir)
        
        # 检查源数据
        if not (data_path / "labeled_dataset.json").exists():
            self.checks_failed.append(f"✗ 缺少 {data_dir}/labeled_dataset.json")
            return False
        else:
            file_size = (data_path / "labeled_dataset.json").stat().st_size / 1e6
            self.checks_passed.append(f"✓ 数据集存在 ({file_size:.2f} MB)")
        
        # 检查分割后的数据
        splits = ["train.json", "val.json", "test.json"]
        has_splits = all((data_path / split).exists() for split in splits)
        
        if has_splits:
            sizes = {split: (data_path / split).stat().st_size / 1e6 for split in splits}
            status = f"✓ 已分割数据集: train={sizes['train.json']:.2f}MB, val={sizes['val.json']:.2f}MB, test={sizes['test.json']:.2f}MB"
            self.checks_passed.append(status)
        else:
            self.warnings.append("⚠ 数据未分割，需要运行: python code_module/data/split_dataset.py")
        
        return True
    
    def check_model_files(self, model_dir: str = "code_module/models") -> bool:
        """检查模型文件完整性"""
        model_path = Path(model_dir)
        
        required_files = ["codebert_detector.py", "__init__.py", "config.py"]
        missing = [f for f in required_files if not (model_path / f).exists()]
        
        if missing:
            self.checks_failed.append(f"✗ 缺少模型文件: {', '.join(missing)}")
            return False
        else:
            self.checks_passed.append(f"✓ 模型文件完整")
            return True
    
    def check_training_scripts(self, train_dir: str = "code_module/training") -> bool:
        """检查训练脚本完整性"""
        train_path = Path(train_dir)
        
        required_files = ["train_code.py", "losses.py", "metrics.py", "dataset.py"]
        missing = [f for f in required_files if not (train_path / f).exists()]
        
        if missing:
            self.checks_failed.append(f"✗ 缺少训练脚本: {', '.join(missing)}")
            return False
        else:
            self.checks_passed.append(f"✓ 训练脚本完整")
            return True
    
    def check_disk_space(self, min_gb: float = 20) -> bool:
        """检查磁盘空间"""
        import shutil
        usage = shutil.disk_usage("/")
        free_gb = usage.free / 1e9
        
        if free_gb >= min_gb:
            self.checks_passed.append(f"✓ 磁盘空间充足 ({free_gb:.1f} GB 可用)")
            return True
        else:
            self.checks_failed.append(f"✗ 磁盘空间不足 ({free_gb:.1f} GB，需要 {min_gb} GB)")
            return False
    
    def print_report(self):
        """打印检查报告"""
        print("\n" + "="*60)
        print("           云端环境检查报告")
        print("="*60 + "\n")
        
        # 通过的检查
        if self.checks_passed:
            print("✅ 通过的检查:")
            for check in self.checks_passed:
                print(f"  {check}")
        
        # 失败的检查
        if self.checks_failed:
            print("\n❌ 失败的检查:")
            for check in self.checks_failed:
                print(f"  {check}")
        
        # 警告
        if self.warnings:
            print("\n⚠️  警告:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        # 总结
        print("\n" + "-"*60)
        total = len(self.checks_passed) + len(self.checks_failed)
        passed = len(self.checks_passed)
        print(f"检查结果: {passed}/{total} 通过\n")
        
        if not self.checks_failed:
            print("🎉 环境检查全部通过，可以开始训练了！")
            print("\n建议的下一步:")
            print("  1. 运行数据分割: python code_module/data/split_dataset.py")
            print("  2. 开始训练: python code_module/training/train_code.py \\")
            print("               --train-data code_module/data/train.json \\")
            print("               --val-data code_module/data/val.json")
            return True
        else:
            print("❌ 环境检查失败，请解决上述问题后重试")
            return False
        
        print("="*60 + "\n")


def run_full_check():
    """执行完整检查"""
    checker = EnvironmentChecker()
    
    # 基础检查
    checker.check_python_version()
    
    # 依赖检查
    libs = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("scikit-learn", "sklearn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
    ]
    
    for lib_name, import_name in libs:
        checker.check_library(lib_name, import_name)
    
    # GPU 检查
    checker.check_gpu()
    
    # 文件检查
    checker.check_data_files()
    checker.check_model_files()
    checker.check_training_scripts()
    
    # 资源检查
    checker.check_disk_space()
    
    # 打印报告
    success = checker.print_report()
    
    return success


if __name__ == "__main__":
    success = run_full_check()
    sys.exit(0 if success else 1)
