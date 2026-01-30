"""
CodeBERT 恶意代码检测器 - 推理脚本

功能：
1. 加载训练好的模型
2. 对代码进行风险评估
3. 返回置信度和不确定性估计
4. 支持批量推理
5. 生成详细的推理报告
"""
import torch
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from transformers import RobertaTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODEL_NAME, MAX_LENGTH, DEVICE, TYPE_LABELS, TYPE_TO_ID,
    InferenceConfig
)
from models.codebert_detector import CodeBertDetector


class CodeBertInferencer:
    """推理器"""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        model_name: str = MODEL_NAME,
        device: torch.device = DEVICE,
        config: InferenceConfig = InferenceConfig()
    ):
        """
        Args:
            model_path: 保存的模型目录路径
            model_name: 预训练模型名称
            device: 计算设备
            config: 推理配置
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.device = device
        self.config = config
        
        # 加载模型和 tokenizer
        print(f"⏳ 加载模型: {model_path}")
        self._load_model()
        
        print(f"⏳ 加载 tokenizer: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        self.model.eval()
        print(f"✅ 模型加载完成")
    
    def _load_model(self):
        """加载模型"""
        # 支持直接传入文件或目录；统一使用 best.pt（与训练保存一致）
        candidate_path = self.model_path
        if candidate_path.is_dir():
            candidate_path = candidate_path / "best.pt"

        if candidate_path.exists():
            self.model = CodeBertDetector()
            state_dict = torch.load(candidate_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"未找到模型文件: {candidate_path}")
        
        self.model.to(self.device)
    
    @torch.no_grad()
    def infer_single(self, code: str) -> Dict:
        """
        对单个代码样本进行推理
        
        Args:
            code: 代码字符串
        
        Returns:
            推理结果字典：
            {
                'code': str,                    # 输入代码（截断后）
                'is_malicious': bool,           # 是否为恶意代码
                'binary_score': float,          # P(malicious)，[0, 1]
                'binary_scores': dict,          # {'normal': float, 'malicious': float}
                'type_label': str,              # 恶意类型
                'type_scores': dict,            # 各恶意类型的概率
                'certainty': str,               # 'high', 'medium', 'low'
                'confidence': float,            # 最高概率值
                'margin': float,                # 最高概率 - 次高概率
                'decision_uncertainty': str,    # 'confident', 'uncertain', 'reject'
            }
        """
        # Tokenization
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 推理
        outputs = self.model(input_ids, attention_mask)
        binary_logits = outputs['binary_logits']
        type_logits = outputs['type_logits']
        
        # 二分类概率
        binary_probs = torch.softmax(binary_logits, dim=1)[0].cpu().numpy()
        binary_pred = np.argmax(binary_logits[0].cpu().numpy())
        
        # 类型分类概率
        type_probs = torch.softmax(type_logits, dim=1)[0].cpu().numpy()
        type_pred = np.argmax(type_logits[0].cpu().numpy())
        
        # 计算置信度和 margin
        binary_scores_sorted = np.sort(binary_probs)[::-1]
        binary_confidence = binary_scores_sorted[0]
        binary_margin = binary_scores_sorted[0] - binary_scores_sorted[1]
        
        type_scores_sorted = np.sort(type_probs)[::-1]
        type_confidence = type_scores_sorted[0]
        type_margin = type_scores_sorted[0] - type_scores_sorted[1]
        
        # 判断置信度等级
        if binary_confidence > self.config.high_confidence_threshold:
            certainty = 'high'
        elif binary_confidence > self.config.medium_confidence_threshold:
            certainty = 'medium'
        else:
            certainty = 'low'
        
        # 判断决策不确定性
        is_malicious = binary_pred == 1
        malicious_prob = binary_probs[1]
        
        if malicious_prob < self.config.uncertainty_threshold_low:
            decision_uncertainty = 'reject'
        elif malicious_prob > self.config.uncertainty_threshold_high:
            decision_uncertainty = 'confident'
        else:
            decision_uncertainty = 'uncertain'
        
        # 类型标签
        type_label_name = TYPE_LABELS.get(type_pred, f'Unknown({type_pred})')
        
        # 构造结果
        result = {
            'code': code[:200],  # 截断显示
            'is_malicious': bool(is_malicious),
            'binary_score': float(malicious_prob),
            'binary_scores': {
                'normal': float(binary_probs[0]),
                'malicious': float(binary_probs[1])
            },
            'type_label': type_label_name,
            'type_scores': {
                TYPE_LABELS.get(i, f'Unknown({i})'): float(type_probs[i])
                for i in range(len(type_probs))
            },
            'certainty': certainty,
            'confidence': float(binary_confidence),
            'margin': float(binary_margin),
            'decision_uncertainty': decision_uncertainty,
            'type_confidence': float(type_confidence),
            'type_margin': float(type_margin)
        }
        
        return result
    
    @torch.no_grad()
    def infer_batch(self, codes: List[str], batch_size: int = 32) -> List[Dict]:
        """
        对一批代码样本进行推理
        
        Args:
            codes: 代码列表
            batch_size: 批处理大小
        
        Returns:
            推理结果列表
        """
        results = []
        
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i+batch_size]
            
            for code in batch_codes:
                result = self.infer_single(code)
                results.append(result)
        
        return results
    
    def infer_from_file(self, file_path: Union[str, Path]) -> Dict:
        """
        从文件读取代码进行推理
        
        Args:
            file_path: 代码文件路径
        
        Returns:
            推理结果
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        result = self.infer_single(code)
        result['file_path'] = str(file_path)
        
        return result
    
    def format_result(self, result: Dict, verbose: bool = True) -> str:
        """
        格式化推理结果为可读字符串
        
        Args:
            result: 推理结果字典
            verbose: 是否详细输出
        
        Returns:
            格式化字符串
        """
        output = []
        output.append("=" * 70)
        output.append("CodeBERT 恶意代码检测结果")
        output.append("=" * 70)
        
        # 基本判断
        status = "❌ 恶意代码" if result['is_malicious'] else "✅ 安全代码"
        output.append(f"\n判断结果: {status}")
        output.append(f"恶意概率: {result['binary_score']:.4f}")
        output.append(f"置信度: {result['certainty'].upper()}")
        
        # 决策不确定性
        if result['decision_uncertainty'] == 'uncertain':
            output.append(f"⚠️  该样本处于不确定区间，建议人工审核")
        
        # 恶意类型
        output.append(f"\n恶意类型: {result['type_label']}")
        output.append(f"类型概率: {result['type_confidence']:.4f}")
        
        if verbose:
            output.append(f"\n二分类概率分布:")
            for label, prob in result['binary_scores'].items():
                bar_length = int(prob * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                output.append(f"  {label:10s}: {bar} {prob:.4f}")
            
            output.append(f"\n各类型概率分布:")
            top_types = sorted(
                result['type_scores'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # 只显示 top 5
            
            for type_name, prob in top_types:
                bar_length = int(prob * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                output.append(f"  {type_name:20s}: {bar} {prob:.4f}")
        
        output.append("\n" + "=" * 70)
        
        return "\n".join(output)


def main():
    """交互式推理主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CodeBERT 恶意代码检测器推理")
    parser.add_argument(
        "--model-path", "-mp",
        type=Path,
        required=True,
        help="保存的模型路径"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="输入代码文件路径"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=['single', 'file', 'interactive'],
        default='interactive',
        help="推理模式"
    )
    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = CodeBertInferencer(args.model_path)
    
    if args.mode == 'file' and args.input:
        # 文件模式
        result = inferencer.infer_from_file(args.input)
        output = inferencer.format_result(result, verbose=args.verbose)
        print(output)
    
    elif args.mode == 'interactive':
        # 交互模式
        print("\n" + "=" * 70)
        print("CodeBERT 恶意代码检测器 - 交互模式")
        print("=" * 70)
        print("请输入代码（多行可用 ''' 包围，输入 'quit' 退出）：\n")
        
        while True:
            code_input = input(">> ")
            
            if code_input.lower() == 'quit':
                print("👋 再见！")
                break
            
            if not code_input.strip():
                continue
            
            # 推理
            result = inferencer.infer_single(code_input)
            output = inferencer.format_result(result, verbose=args.verbose)
            print(output)
    
    else:
        print("❌ 请指定有效的模式和输入")


if __name__ == "__main__":
    main()
