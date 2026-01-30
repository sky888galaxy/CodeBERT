"""定义流量特征规范。

这是一个独立的流量检测子项目。如果未来需要与 codebert 项目协同，
可以通过统一的 DetectionResult 结构对接，这里的特征规范可直接复用。
"""
from dataclasses import dataclass
from typing import List

# 规范版本，便于后续 checkpoint/推理时校验特征顺序一致性。
SPEC_VERSION = "v1"

# 模型设计的通用流量特征空间。按需扩展或裁剪，但保持顺序稳定以防训练/推理错位。
CORE_FLOW_FEATURES: List[str] = [
    "frame_len",
    "ip_hdr_len",
    "ip_len",
    "ip_flags_df",
    "ip_flags_mf",
    "ip_ttl",
    "ip_proto",
    "src_port",
    "dst_port",
    "tcp_len",
    "tcp_window_size",
    "tcp_flag_res",
    "tcp_flag_ns",
    "tcp_flag_cwr",
    "tcp_flag_ecn",
    "tcp_flag_urg",
    "tcp_flag_ack",
    "tcp_flag_psh",
    "tcp_flag_rst",
    "tcp_flag_syn",
    "tcp_flag_fin",
    "tcp_time_delta",
    "pkt_per_flow",
    "bytes_per_flow",
    "duration",
    "fwd_pkt_count",
    "bwd_pkt_count",
    # 流内包数量
    "flow_pkt_count",
    # 流内总字节数（frame.len 求和）
    "flow_bytes_sum",
    # 流内平均包长度（frame.len 均值）
    "flow_bytes_mean",
    # 流内 tcp.time_delta 均值
    "time_delta_mean",
    # 流内 tcp.time_delta 标准差，单包流置为 0
    "time_delta_std",
    # 流内 SYN 报文数量
    "syn_count",
    # 流内 ACK 报文数量
    "ack_count",
    # 流内 RST 报文数量
    "rst_count",
    # 流内 FIN 报文数量
    "fin_count",
    # 当前报文在流内的顺序（从 1 开始）
    "pkt_index_in_flow",
    # 报文在流内的归一化位置（index/flow_pkt_count）
    "pkt_index_norm",
]


@dataclass
class FlowFeatureSpec:
    """流量特征规格定义。

    Attributes:
        feature_names: 按顺序排列的特征名列表。
        input_dim: 输入向量维度，等于特征数量。
    """

    feature_names: List[str]
    input_dim: int


def get_default_flow_feature_spec() -> FlowFeatureSpec:
    """返回默认的流量特征规格。

    使用 CORE_FLOW_FEATURES 构造输入维度，后续模型、预处理和推理统一引用。
    """

    return FlowFeatureSpec(feature_names=CORE_FLOW_FEATURES, input_dim=len(CORE_FLOW_FEATURES))


__all__ = [
    "CORE_FLOW_FEATURES",
    "FlowFeatureSpec",
    "get_default_flow_feature_spec",
    "SPEC_VERSION",
]
