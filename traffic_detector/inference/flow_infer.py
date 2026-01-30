"""流量检测推理数据结构。

本文件定义推理输入输出的数据类。项目是独立的流量检测子系统，
若未来与 codebert 项目整合，可通过 DetectionResult 作为统一的协同接口。
"""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class FlowEvent:
    """用于推理的一条流量事件。"""

    event_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    features: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """模型输出的检测结果。

    如果与 codebert 项目协同，可直接返回此结构，便于统一消费与决策。
    """

    event_id: str
    is_malicious: bool
    attack_type: str
    confidence: float
    suggested_action: str
    raw_scores: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


__all__ = ["FlowEvent", "DetectionResult"]
