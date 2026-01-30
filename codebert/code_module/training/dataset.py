"""
数据集加载和处理模块

支持：
1. JSON 格式数据加载
2. CodeBERT 分词
3. 强/弱标注样本区分
4. 样本权重计算
5. 批处理和 DataLoader
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import sys
from pathlib import Path as PathlibPath

# 添加上级目录到 path
sys.path.insert(0, str(PathlibPath(__file__).parent.parent))
from config import MODEL_NAME, MAX_LENGTH, TYPE_LABELS, TYPE_TO_ID


class MaliciousCodeDataset(Dataset):
    """
    恶意代码数据集
    
    支持：
    - 强/弱标注样本区分
    - 多任务学习（二分类 + 类型分类）
    - 自动 tokenization
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = MAX_LENGTH,
        task: str = "multitask"  # "binary" 或 "multitask"
    ):
        """
        Args:
            data: 数据列表，每条格式：
                {
                    "code": "...",
                    "binary_label": 0/1,
                    "type_label": 0-5 (可选),
                    "label_type": "strong" / "weak" (可选，默认 "strong"),
                    ...
                }
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
            task: "binary" (只进行二分类) 或 "multitask" (二分类+类型分类)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        
        # 数据验证与标准化
        self._validate_and_normalize()
    
    def _validate_and_normalize(self):
        """数据验证与标准化处理"""
        normalized_data = []
        
        for idx, sample in enumerate(self.data):
            # 提取必要字段
            code = sample.get('code', '')
            if not code or not isinstance(code, str):
                print(f"⚠️  样本 {idx} 代码为空，跳过")
                continue
            
            # 二分类标签（必须）
            binary_label = sample.get('binary_label')
            if binary_label is None:
                # 尝试从 'label' 字段获取（向后兼容）
                binary_label = sample.get('label')
            
            if binary_label is None:
                print(f"⚠️  样本 {idx} 缺少 binary_label，跳过")
                continue
            
            binary_label = int(binary_label)
            
            # 类型标签（可选）
            type_label = sample.get('type_label')
            if type_label is None:
                # 默认根据 binary_label 设置 type_label
                # binary_label=0 -> type_label=0 (Normal)
                # binary_label=1 -> type_label=未知，标记为 -1
                type_label = 0 if binary_label == 0 else -1
            else:
                type_label = int(type_label)
            
            # 标注类型（强/弱）
            label_type = sample.get('label_type', 'strong')
            if label_type not in ['strong', 'weak']:
                label_type = 'strong'
            
            # 构建规范化样本
            normalized_sample = {
                'code': code,
                'binary_label': binary_label,
                'type_label': type_label,
                'label_type': label_type,
                'source': sample.get('source', 'unknown')
            }
            
            normalized_data.append(normalized_sample)
        
        self.data = normalized_data
        print(f"✅ 数据加载完成: 共 {len(self.data)} 个有效样本")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            dict 包含：
                - input_ids: [max_length]
                - attention_mask: [max_length]
                - binary_label: int
                - type_label: int
                - label_type: str
                - type_mask: bool (是否有有效 type_label)
                - sample_weight: float (强/弱标注权重)
        """
        sample = self.data[idx]
        code = sample['code']
        binary_label = sample['binary_label']
        type_label = sample['type_label']
        label_type = sample['label_type']
        
        # Tokenization
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 计算 type_mask（type_label 是否有效）
        # 约定：type_label = -1 表示无效；保持原值以便后续过滤
        type_mask = 1 if type_label >= 0 else 0
        
        # 计算样本权重（强/弱标注）
        sample_weight = 1.0 if label_type == 'strong' else 0.4
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'binary_label': torch.tensor(binary_label, dtype=torch.long),
            'type_label': torch.tensor(type_label, dtype=torch.long),
            'label_type': label_type,
            'type_mask': torch.tensor(type_mask, dtype=torch.bool),
            'sample_weight': torch.tensor(sample_weight, dtype=torch.float32)
        }
        
        return result


def load_data_from_json(
    json_path: Union[str, Path],
    limit: Optional[int] = None
) -> List[Dict]:
    """
    从 JSON 文件加载数据
    
    Args:
        json_path: JSON 文件路径
        limit: 最多加载的样本数（None 表示全部）
    
    Returns:
        数据列表
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    print(f"✅ 从 {json_path} 加载 {len(data)} 条数据")
    
    return data


def create_data_loaders(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: Optional[List[Dict]] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True,
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
    task: str = "multitask"
) -> Dict[str, DataLoader]:
    """
    创建训练、验证、测试 DataLoader
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据（可选）
        batch_size: batch 大小
        num_workers: DataLoader workers 数
        shuffle_train: 是否对训练数据进行 shuffle
        model_name: tokenizer 模型名称
        max_length: 最大序列长度
        task: "binary" 或 "multitask"
    
    Returns:
        DataLoaders 字典
    """
    # 加载 tokenizer
    print(f"⏳ 加载 tokenizer: {model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    print(f"✅ tokenizer 加载完成")
    
    # 创建 Dataset
    train_dataset = MaliciousCodeDataset(train_data, tokenizer, max_length, task)
    val_dataset = MaliciousCodeDataset(val_data, tokenizer, max_length, task)
    
    # 创建 DataLoader
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    if test_data is not None:
        test_dataset = MaliciousCodeDataset(test_data, tokenizer, max_length, task)
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    print(f"✅ DataLoaders 创建完成")
    print(f"   - 训练集: {len(train_dataset)} 样本")
    print(f"   - 验证集: {len(val_dataset)} 样本")
    if test_data is not None:
        print(f"   - 测试集: {len(test_dataset)} 样本")
    
    return loaders


def collate_fn_multitask(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数（用于多任务学习）
    
    将多个样本组合为一个 batch，处理可变长度的序列
    """
    # 获取 batch 中所有项
    input_ids_list = []
    attention_mask_list = []
    binary_labels = []
    type_labels = []
    type_masks = []
    sample_weights = []
    
    for item in batch:
        input_ids_list.append(item['input_ids'])
        attention_mask_list.append(item['attention_mask'])
        binary_labels.append(item['binary_label'])
        type_labels.append(item['type_label'])
        type_masks.append(item['type_mask'])
        sample_weights.append(item['sample_weight'])
    
    # 堆叠为 tensor
    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'binary_label': torch.stack(binary_labels),
        'type_label': torch.stack(type_labels),
        'type_mask': torch.stack(type_masks),
        'sample_weight': torch.stack(sample_weights)
    }


def analyze_dataset_statistics(data: List[Dict]) -> Dict:
    """
    分析数据集统计信息
    
    Returns:
        统计信息字典
    """
    stats = {
        'total_samples': len(data),
        'binary_distribution': {0: 0, 1: 0},
        'type_distribution': {},
        'label_type_distribution': {'strong': 0, 'weak': 0},
        'avg_code_length': 0,
        'code_length_distribution': {'<100': 0, '100-500': 0, '500-1000': 0, '>1000': 0}
    }
    
    code_lengths = []
    
    for sample in data:
        # 二分类分布
        binary_label = sample.get('binary_label')
        if binary_label is not None:
            stats['binary_distribution'][int(binary_label)] += 1
        
        # 类型分布
        type_label = sample.get('type_label')
        if type_label is not None:
            type_label_str = TYPE_LABELS.get(int(type_label), f"Unknown({type_label})")
            stats['type_distribution'][type_label_str] = stats['type_distribution'].get(type_label_str, 0) + 1
        
        # 标注类型分布
        label_type = sample.get('label_type', 'strong')
        stats['label_type_distribution'][label_type] += 1
        
        # 代码长度
        code = sample.get('code', '')
        code_len = len(code)
        code_lengths.append(code_len)
        
        if code_len < 100:
            stats['code_length_distribution']['<100'] += 1
        elif code_len < 500:
            stats['code_length_distribution']['100-500'] += 1
        elif code_len < 1000:
            stats['code_length_distribution']['500-1000'] += 1
        else:
            stats['code_length_distribution']['>1000'] += 1
    
    if code_lengths:
        stats['avg_code_length'] = sum(code_lengths) / len(code_lengths)
        stats['max_code_length'] = max(code_lengths)
        stats['min_code_length'] = min(code_lengths)
    
    return stats


def print_dataset_statistics(stats: Dict) -> None:
    """打印数据集统计信息"""
    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    
    print(f"总样本数: {stats['total_samples']}")
    
    print(f"\n二分类分布:")
    for label, count in sorted(stats['binary_distribution'].items()):
        label_name = "正常" if label == 0 else "恶意"
        pct = 100.0 * count / stats['total_samples']
        print(f"  {label_name:6s}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n类型分布:")
    for type_label, count in sorted(stats['type_distribution'].items()):
        pct = 100.0 * count / stats['total_samples']
        print(f"  {type_label:20s}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n标注类型分布:")
    for label_type, count in stats['label_type_distribution'].items():
        pct = 100.0 * count / stats['total_samples']
        print(f"  {label_type:10s}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n代码长度统计:")
    print(f"  平均长度: {stats['avg_code_length']:.1f}")
    print(f"  最大长度: {stats['max_code_length']}")
    print(f"  最小长度: {stats['min_code_length']}")
    print(f"  分布:")
    for range_label, count in stats['code_length_distribution'].items():
        pct = 100.0 * count / stats['total_samples']
        print(f"    {range_label:10s}: {count:6d} ({pct:5.1f}%)")
    
    print("=" * 60 + "\n")


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试数据集模块")
    print("=" * 60)
    
    # 创建模拟数据
    test_data = [
        {
            "code": "eval(input())",
            "binary_label": 1,
            "type_label": 2,
            "label_type": "strong",
            "source": "test"
        },
        {
            "code": "print('hello')",
            "binary_label": 0,
            "type_label": 0,
            "label_type": "strong",
            "source": "test"
        },
        {
            "code": "import os; os.system('rm -rf /')",
            "binary_label": 1,
            "type_label": 1,
            "label_type": "weak",
            "source": "test"
        }
    ]
    
    # 测试 Dataset
    print("\n--- 测试 MaliciousCodeDataset ---")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    dataset = MaliciousCodeDataset(test_data, tokenizer)
    print(f"✅ Dataset 创建成功: {len(dataset)} 样本")
    
    sample = dataset[0]
    print(f"\n第一个样本:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  binary_label: {sample['binary_label']}")
    print(f"  type_label: {sample['type_label']}")
    print(f"  type_mask: {sample['type_mask']}")
    print(f"  sample_weight: {sample['sample_weight']}")
    
    # 测试 DataLoader
    print("\n--- 测试 DataLoader ---")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_fn_multitask
    )
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  binary_label: {batch['binary_label']}")
        print(f"  type_label: {batch['type_label']}")
        print(f"  type_mask: {batch['type_mask']}")
        print(f"  sample_weight: {batch['sample_weight']}")
        break
    
    # 测试统计
    print("\n--- 测试数据集统计 ---")
    stats = analyze_dataset_statistics(test_data)
    print_dataset_statistics(stats)
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
