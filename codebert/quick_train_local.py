"""
本地快速训练脚本 - RTX 4060 优化版
使用小批量数据快速验证模型训练流程
"""
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.insert(0, str(Path(__file__).parent))
from code_module.models.codebert_detector import CodeBertDetector
from code_module.training.losses import MultiTaskLoss, compute_class_weights
from code_module.training.metrics import MetricsComputer

print("=" * 70)
print("🚀 CodeBERT 本地快速训练 - RTX 4060 热身版")
print("=" * 70)

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✓ 设备: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 训练配置（针对 RTX 4060 8GB 优化）
CONFIG = {
    'batch_size': 8,        # 小批量避免显存溢出
    'epochs': 5,            # 快速验证
    'learning_rate': 2e-5,
    'max_length': 256,      # 缩短序列长度
    'num_workers': 0,       # Windows 上避免多进程问题
}

print(f"\n✓ 训练配置:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# 简化的数据集类
class SimpleCodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        code = item.get('code', '')
        label = item.get('label', item.get('binary_label', 0))
        
        # Tokenize
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'binary_label': torch.tensor(label, dtype=torch.long),
            'type_label': torch.tensor(0, dtype=torch.long),  # 简化：都设为0
        }

# 加载数据
print(f"\n✓ 加载数据...")
data_file = Path('code_module/data/small_sample.json')
with open(data_file) as f:
    data = json.load(f)

# 分割数据 (160训练 + 40验证)
train_data = data[:160]
val_data = data[160:]
print(f"  训练集: {len(train_data)} 样本")
print(f"  验证集: {len(val_data)} 样本")

# 加载 tokenizer
print(f"\n✓ 加载 CodeBERT tokenizer (offline)...")
tokenizer = AutoTokenizer.from_pretrained("./models/codebert-base")

# 创建数据集
train_dataset = SimpleCodeDataset(train_data, tokenizer, CONFIG['max_length'])
val_dataset = SimpleCodeDataset(val_data, tokenizer, CONFIG['max_length'])

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True,
    num_workers=CONFIG['num_workers']
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers']
)

# 创建模型
print(f"\n✓ 创建模型 (offline)...")
model = CodeBertDetector(
    model_name="./models/codebert-base",
    num_types=6,  # 6种类型
    dropout_prob=0.1
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数: {total_params:,}")
print(f"  可训练: {trainable_params:,}")

# 计算类别权重
labels = [d.get('label', d.get('binary_label', 0)) for d in train_data]
binary_weights = compute_class_weights(labels, num_classes=2)
print(f"\n✓ 类别权重: {binary_weights}")

# 损失函数和优化器
criterion = MultiTaskLoss(
    binary_class_weights=binary_weights,
    use_focal_loss=True,
    focal_gamma=2.0
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

# 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        binary_labels = batch['binary_label'].to(device)
        type_labels = batch['type_label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        binary_logits = outputs['binary_logits']
        type_logits = outputs['type_logits']
        
        # 计算损失
        type_mask = (type_labels >= 0)  # 简单的mask
        loss_dict = criterion(
            binary_logits, binary_labels,
            type_logits, type_labels, type_mask
        )
        
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集预测
        preds = torch.argmax(binary_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(binary_labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    
    # 计算准确率
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)
    
    return avg_loss, accuracy

# 验证函数
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        binary_labels = batch['binary_label'].to(device)
        type_labels = batch['type_label'].to(device)
        
        outputs = model(input_ids, attention_mask)
        binary_logits = outputs['binary_logits']
        type_logits = outputs['type_logits']
        
        type_mask = (type_labels >= 0)
        loss_dict = criterion(
            binary_logits, binary_labels,
            type_logits, type_labels, type_mask
        )
        
        total_loss += loss_dict['total_loss'].item()
        
        preds = torch.argmax(binary_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(binary_labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)
    
    return avg_loss, accuracy

# 开始训练
print(f"\n{'=' * 70}")
print("🔥 开始训练...")
print(f"{'=' * 70}\n")

best_val_acc = 0
for epoch in range(CONFIG['epochs']):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
    print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
    print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # 保存最佳模型
        save_dir = Path('experiments/quick_train')
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / 'best_model.pt')
        print(f"  ✓ 保存最佳模型 (val_acc: {val_acc:.2%})")
    print()

print(f"{'=' * 70}")
print(f"✅ 训练完成！")
print(f"  最佳验证准确率: {best_val_acc:.2%}")
print(f"  模型已保存到: experiments/quick_train/best_model.pt")
print(f"{'=' * 70}")

# 显示 GPU 显存使用情况
if torch.cuda.is_available():
    print(f"\n📊 GPU 显存使用:")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  峰值: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
