"""
CodeBERT 多任务检测器（离线、本地模型加载）。
结构：CodeBERT encoder -> Dropout -> binary/type 线性头。
"""
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import LOCAL_MODEL_PATH, NUM_TYPES, TYPE_LABELS, TYPE_TO_ID, MAX_LENGTH

TYPE_ID_TO_NAME = TYPE_LABELS
TYPE_NAME_TO_ID = TYPE_TO_ID


def load_local_tokenizer(model_path=LOCAL_MODEL_PATH):
    """Helper to load tokenizer from local path (offline)."""
    return AutoTokenizer.from_pretrained(str(model_path))


class CodeBertDetector(nn.Module):
    """
    CodeBERT 多任务检测器
    
    结构：
    - CodeBERT encoder
    - Dropout
    - Binary classification head (2 classes)
    - Type classification head (NUM_TYPES classes)
    """
    
    def __init__(
        self,
        model_path=LOCAL_MODEL_PATH,
        num_types: int = NUM_TYPES,
        dropout_prob: float = 0.1,
        freeze_encoder: bool = False,
        max_length: int = MAX_LENGTH,
        load_tokenizer: bool = True,
    ):
        super().__init__()

        self.model_path = str(model_path)
        self.num_types = num_types
        self.max_length = max_length

        # 本地加载 CodeBERT 编码器（离线）
        self.encoder = AutoModel.from_pretrained(self.model_path)
        hidden_size = self.encoder.config.hidden_size

        # 本地 tokenizer（可选，便于推理）
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path) if load_tokenizer else None

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_prob)
        self.binary_head = nn.Linear(hidden_size, 2)
        self.type_head = nn.Linear(hidden_size, num_types)

        self._init_weights(self.binary_head)
        self._init_weights(self.type_head)
    
    def _init_weights(self, module):
        """初始化权重（遵循 BERT 的初始化策略）"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask, return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_hidden: 是否返回隐藏层表示（用于可视化）
        
        Returns:
            dict: {
                "binary_logits": [batch_size, 2],
                "type_logits": [batch_size, num_types],
                "hidden": [batch_size, hidden_size]  # 可选
            }
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        binary_logits = self.binary_head(pooled_output)
        type_logits = self.type_head(pooled_output)

        result = {
            "binary_logits": binary_logits,
            "type_logits": type_logits,
        }
        if return_hidden:
            result["hidden"] = pooled_output
        return result
    
    def predict(self, input_ids, attention_mask, return_probs=True):
        """
        推理模式（返回概率而不是 logits）
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_probs: 是否返回概率（否则返回 logits）
        
        Returns:
            dict: {
                "binary_probs": [batch_size, 2],  # or logits
                "type_probs": [batch_size, num_types],
                "binary_pred": [batch_size],  # 预测标签
                "type_pred": [batch_size]
            }
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            binary_logits = outputs["binary_logits"]
            type_logits = outputs["type_logits"]
            
            if return_probs:
                binary_probs = torch.softmax(binary_logits, dim=1)
                type_probs = torch.softmax(type_logits, dim=1)
            else:
                binary_probs = binary_logits
                type_probs = type_logits
            
            binary_pred = torch.argmax(binary_logits, dim=1)
            type_pred = torch.argmax(type_logits, dim=1)
            
            return {
                "binary_probs": binary_probs,
                "type_probs": type_probs,
                "binary_pred": binary_pred,
                "type_pred": type_pred
            }

    def encode_code(self, code_str: str, max_length: int = None, device: str = None) -> Dict[str, torch.Tensor]:
        """将原始代码字符串编码为模型输入（离线，本地 tokenizer）。"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Instantiate with load_tokenizer=True or call load_local_tokenizer().")

        max_len = max_length or self.max_length
        encoded = self.tokenizer(
            code_str,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        if device:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded
    
    def freeze_encoder(self):
        """冻结 encoder 参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """解冻 encoder 参数"""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_num_params(self):
        """获取模型参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
    
    def get_encoder_output(self, input_ids, attention_mask):
        """
        获取 encoder 的输出（用于特征提取）
        
        Returns:
            encoder_output: [batch_size, seq_len, hidden_size]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state
    
    def get_pooled_representation(self, input_ids, attention_mask):
        """
        获取 pooled representation（[CLS] token）
        
        Returns:
            pooled: [batch_size, hidden_size]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state[:, 0, :]
    
    def compute_attention_weights(self, input_ids, attention_mask):
        """
        获取模型的注意力权重（用于可视化）
        
        Returns:
            attentions: tuple of attention matrices from each layer
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=True
        )
        return outputs.attentions


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("测试 CodeBertDetector")
    print("=" * 50)
    
    # 创建模型
    model = CodeBertDetector()
    print(f"✅ 模型创建成功")
    print(f"   参数量: {model.get_num_params()}")
    
    # 测试前向传播
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = model(input_ids, attention_mask, return_hidden=True)
    print(f"\n✅ 前向传播成功")
    print(f"   binary_logits shape: {outputs['binary_logits'].shape}")
    print(f"   type_logits shape: {outputs['type_logits'].shape}")
    print(f"   hidden shape: {outputs['hidden'].shape}")
    
    # 测试推理模式
    preds = model.predict(input_ids, attention_mask)
    print(f"\n✅ 推理模式成功")
    print(f"   binary_probs shape: {preds['binary_probs'].shape}")
    print(f"   binary_pred: {preds['binary_pred']}")
    print(f"   type_pred: {preds['type_pred']}")
    
    # 测试冻结/解冻
    model.freeze_encoder()
    print(f"\n✅ 冻结 encoder 后参数量: {model.get_num_params()}")
    
    model.unfreeze_encoder()
    print(f"✅ 解冻 encoder 后参数量: {model.get_num_params()}")
