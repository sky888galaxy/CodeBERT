"""通用流量特征转换器。

本模块按 *模型既定的特征空间* 来适配数据集，而不是为某个数据集改模型特征。
提供从原始 CSV 转成统一特征向量的工具，便于训练、评估与推理复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from traffic_detector.features.flow_schema import (
    CORE_FLOW_FEATURES,
    FlowFeatureSpec,
    get_default_flow_feature_spec,
)
from traffic_detector.inference.flow_infer import FlowEvent


@dataclass
class FlowFeatureTransformer:
    """为流量检测模型准备的通用特征转换器。

    设计理念：按照模型定义的特征空间 (CORE_FLOW_FEATURES) 适配数据集，
    而不是反过来修改模型去迎合某个数据集。这样可保持训练/推理一致性，
    也方便未来与 codebert 项目通过 DetectionResult 协同。
    """

    feature_spec: FlowFeatureSpec 
    scaler: Optional[StandardScaler] = None

    def __post_init__(self) -> None:
        # 如果未显式传入，使用默认的特征规范。
        if self.feature_spec is None:
            self.feature_spec = get_default_flow_feature_spec()

    def _append_flow_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """为 DataFrame 增补流级与时序特征。"""

        # 以流为粒度构建上下文信息
        flow_id_cols = ["ip.src", "ip.dst", "tcp.srcport", "tcp.dstport", "ip.proto"]
        df = df.copy()
        df["flow_id"] = df[flow_id_cols].astype(str).agg("-".join, axis=1)

        # 基础列先标准化为数值，方便聚合
        for num_col in ["frame.len", "tcp.time_delta"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0.0)
            else:
                df[num_col] = 0.0

        # TCP flags 统一为 0/1 数值，缺失列补 0
        flag_cols = ["tcp.flags.syn", "tcp.flags.ack", "tcp.flags.reset", "tcp.flags.fin"]
        for col in flag_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            else:
                df[col] = 0

        group = df.groupby("flow_id")

        # 流级包数和字节统计
        df["flow_pkt_count"] = group["frame.len"].transform("count")
        df["flow_bytes_sum"] = group["frame.len"].transform("sum")
        df["flow_bytes_mean"] = group["frame.len"].transform("mean")

        # 流级时间间隔统计
        df["time_delta_mean"] = group["tcp.time_delta"].transform("mean")
        df["time_delta_std"] = group["tcp.time_delta"].transform("std").fillna(0.0)

        # 流级 TCP 标志位计数
        df["syn_count"] = group["tcp.flags.syn"].transform("sum")
        df["ack_count"] = group["tcp.flags.ack"].transform("sum")
        df["rst_count"] = group["tcp.flags.reset"].transform("sum")
        df["fin_count"] = group["tcp.flags.fin"].transform("sum")

        # 位置特征：包在流中的序号与归一化位置
        df["pkt_index_in_flow"] = group.cumcount() + 1
        df["pkt_index_norm"] = df["pkt_index_in_flow"] / df["flow_pkt_count"].replace(0, 1)

        return df

    def _map_columns_to_features(self, df: pd.DataFrame) -> np.ndarray:
        """将 DataFrame 按 CORE_FLOW_FEATURES 顺序映射为数值矩阵 (未经归一化)。

        对于当前数据集中缺失的预留特征（如 pkt_per_flow、duration 等），统一填 0，
        这些特征未来用于流级聚合统计时再填充。
        """

        col_map: Dict[str, str] = {
            "frame.len": "frame_len",
            "ip.hdr_len": "ip_hdr_len",
            "ip.len": "ip_len",
            "ip.flags.df": "ip_flags_df",
            "ip.flags.mf": "ip_flags_mf",
            "ip.ttl": "ip_ttl",
            "ip.proto": "ip_proto",
            "tcp.srcport": "src_port",
            "tcp.dstport": "dst_port",
            "tcp.len": "tcp_len",
            "tcp.window_size": "tcp_window_size",
            "tcp.flags.res": "tcp_flag_res",
            "tcp.flags.ns": "tcp_flag_ns",
            "tcp.flags.cwr": "tcp_flag_cwr",
            "tcp.flags.ecn": "tcp_flag_ecn",
            "tcp.flags.urg": "tcp_flag_urg",
            "tcp.flags.ack": "tcp_flag_ack",
            "tcp.flags.push": "tcp_flag_psh",
            "tcp.flags.reset": "tcp_flag_rst",
            "tcp.flags.syn": "tcp_flag_syn",
            "tcp.flags.fin": "tcp_flag_fin",
            "tcp.time_delta": "tcp_time_delta",
            # 流级统计与时序特征，前置计算后直接按同名列映射
            "flow_pkt_count": "flow_pkt_count",
            "flow_bytes_sum": "flow_bytes_sum",
            "flow_bytes_mean": "flow_bytes_mean",
            "time_delta_mean": "time_delta_mean",
            "time_delta_std": "time_delta_std",
            "syn_count": "syn_count",
            "ack_count": "ack_count",
            "rst_count": "rst_count",
            "fin_count": "fin_count",
            "pkt_index_in_flow": "pkt_index_in_flow",
            "pkt_index_norm": "pkt_index_norm",
        }

        # 按 feature_spec.feature_names 的顺序构造矩阵，避免直接依赖 CORE_FLOW_FEATURES 常量。
        feature_order = self.feature_spec.feature_names
        feature_index = {name: idx for idx, name in enumerate(feature_order)}

        # 先初始化全部特征为 0（预留给未来聚合统计特征）。
        features = np.zeros((len(df), len(feature_order)), dtype=np.float64)

        # 对映射到的数据集列逐个填充。
        for csv_col, feature_name in col_map.items():
            if feature_name not in feature_index:
                continue
            if csv_col in df.columns:
                col_values = pd.to_numeric(df[csv_col], errors="coerce").fillna(0.0).to_numpy()
                idx = feature_index[feature_name]
                features[:, idx] = col_values

        # 其他特征如 pkt_per_flow、bytes_per_flow、duration、fwd_pkt_count、bwd_pkt_count 等保留为 0。
        return features.astype(np.float32)

    def fit_transform_from_csv(self, attack_csv_path: str, normal_csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """读取攻击/正常 CSV，生成特征矩阵并拟合 StandardScaler。

        attack -> label 1，normal -> label 0；返回已归一化的 X (float32) 与 y (int64)。
        """

        attack_df = pd.read_csv(attack_csv_path)
        normal_df = pd.read_csv(normal_csv_path)

        print(
            f"[fit_transform] read csv done: attack_shape={attack_df.shape}, normal_shape={normal_df.shape}",
            flush=True,
        )

        attack_df = attack_df.copy()
        normal_df = normal_df.copy()
        attack_df["__label__"] = 1
        normal_df["__label__"] = 0

        merged = pd.concat([attack_df, normal_df], ignore_index=True)
        print(f"[fit_transform] merged shape={merged.shape}, building flow features...", flush=True)
        merged = self._append_flow_context_features(merged)
        print(f"[fit_transform] flow features ready, columns={len(merged.columns)}", flush=True)

        X_raw = self._map_columns_to_features(merged)
        print(f"[fit_transform] mapped to matrix: X_raw shape={X_raw.shape}", flush=True)
        y = merged["__label__"].to_numpy(dtype=np.int64)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw).astype(np.float32)
        print("[fit_transform] scaling done", flush=True)

        return X_scaled, y

    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """将任意包含所需列的 DataFrame 转为特征矩阵并使用已有 scaler 归一化。"""

        if self.scaler is None:
            raise ValueError("scaler 尚未拟合，请先调用 fit_transform_from_csv")

        df_with_flow = self._append_flow_context_features(df)
        X_raw = self._map_columns_to_features(df_with_flow)
        X_scaled = self.scaler.transform(X_raw).astype(np.float32)
        return X_scaled

    def to_flow_events(self, df: pd.DataFrame, is_malicious_label: Optional[int] = None) -> List[FlowEvent]:
        """将 DataFrame 的每一行转换为 FlowEvent。

        features 字段使用“归一化后的特征”，便于直接送入模型；meta 保留原始字段（ip.src、ip.dst 等）。
        如果需要调试原始值，可在 meta 里追加原始未归一化列。
        """

        if self.scaler is None:
            raise ValueError("scaler 尚未拟合，请先调用 fit_transform_from_csv")

        X_raw = self._map_columns_to_features(df)
        X_scaled = self.scaler.transform(X_raw)

        events: List[FlowEvent] = []
        for i, (_, row) in enumerate(df.iterrows()):
            features_dict = {
                name: float(X_scaled[i, idx])
                for idx, name in enumerate(self.feature_spec.feature_names)
            }

            meta = {
                "ip.src": row.get("ip.src"),
                "ip.dst": row.get("ip.dst"),
                "frame.encap_type": row.get("frame.encap_type"),
                "frame.protocols": row.get("frame.protocols"),
                "ip.frag_offset": row.get("ip.frag_offset"),
            }

            events.append(
                FlowEvent(
                    event_id=str(row.name),
                    src_ip=str(row.get("ip.src", "")),
                    dst_ip=str(row.get("ip.dst", "")),
                    src_port=int(row.get("tcp.srcport", 0) or 0),
                    dst_port=int(row.get("tcp.dstport", 0) or 0),
                    protocol=str(row.get("ip.proto", "")),
                    features=features_dict,
                    meta=meta,
                )
            )

        return events


__all__ = ["FlowFeatureTransformer"]
