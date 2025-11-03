from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo.policies import ActorCriticPolicy
from gymnasium import spaces
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool


class QueryFeaturesEmbeddings(nn.Module):
    def __init__(self, embedding_dim, query_window_size, max_columns_per_query, max_edges):
        super(QueryFeaturesEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.query_window_size = query_window_size
        self.max_columns_per_query = max_columns_per_query
        self.max_edges = max_edges
        
        # 边权重嵌入
        self.edge_embedding = nn.Embedding(2, 8)
        
        # 层归一化用于查询嵌入
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # GAT层用于处理查询图结构
        self.query_GAT = GATv2Conv(
            in_channels=embedding_dim, 
            out_channels=embedding_dim, 
            heads=4, 
            edge_dim=8, 
            concat=False,  # 不拼接多头输出
            dropout=0.1
        )
        
        # 后处理网络
        self.gat_ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, query_embeddings, query_maskings, edge_indexes, edge_weights, edge_maskings):
        """
        向量化优化的前向传播，消除双重循环提升性能
        Args:
            query_embeddings: (batch_size, query_window_size, max_columns_per_query, embedding_dim)
            query_maskings: (batch_size, query_window_size, max_columns_per_query)
            edge_indexes: (batch_size, query_window_size, 2, max_edges)
            edge_weights: (batch_size, query_window_size, max_edges)
            edge_maskings: (batch_size, query_window_size, max_edges)
        """
        batch_size = query_embeddings.size(0)
        device = query_embeddings.device
        
        # 对查询嵌入进行层归一化
        query_embeddings = self.layer_norm(query_embeddings)
        
        # 向量化计算所有有效性检查
        valid_nodes_mask = query_maskings.bool()  # (batch_size, query_window_size, max_columns_per_query)
        valid_edges_mask = edge_maskings.bool()   # (batch_size, query_window_size, max_edges)
        
        # 计算每个查询有效节点和边的数量
        valid_nodes_count = valid_nodes_mask.sum(dim=-1)  # (batch_size, query_window_size)
        valid_edges_count = valid_edges_mask.sum(dim=-1)  # (batch_size, query_window_size)
        
        # 找到有效的查询（既有节点又有边）
        valid_queries_mask = (valid_nodes_count > 0) & (valid_edges_count > 0)  # (batch_size, query_window_size)
        valid_queries_indices = torch.nonzero(valid_queries_mask, as_tuple=False)  # (num_valid_queries, 2)非零坐标索引
        
        if valid_queries_indices.size(0) == 0:
            # 没有有效查询，返回零向量
            return torch.zeros(batch_size, self.query_window_size, self.embedding_dim, device=device)
        
        # 批量提取有效查询的数据
        valid_batch_indices = valid_queries_indices[:, 0]
        valid_query_indices = valid_queries_indices[:, 1]
        
        # 提取有效的节点特征和mask
        valid_query_embeddings = query_embeddings[valid_batch_indices, valid_query_indices]  # (num_valid, max_columns, emb_dim)
        valid_node_masks = valid_nodes_mask[valid_batch_indices, valid_query_indices]         # (num_valid, max_columns)
        
        # 提取有效的边数据
        valid_edge_indices = edge_indexes[valid_batch_indices, valid_query_indices]  # (num_valid, 2, max_edges)
        valid_edge_weights = edge_weights[valid_batch_indices, valid_query_indices]  # (num_valid, max_edges)
        valid_edge_masks = valid_edges_mask[valid_batch_indices, valid_query_indices]  # (num_valid, max_edges)

        # ==================== 向量化图批处理开始 ====================
        num_valid_graphs = valid_query_embeddings.size(0)

        # 1. 拼接所有图的节点特征
        all_nodes = valid_query_embeddings[valid_node_masks]  # (total_nodes, emb_dim)
        
        # 2. 创建节点到图的映射索引 (batch index)
        num_nodes_per_graph = valid_node_masks.sum(dim=1)  # (num_valid_graphs,)
        node_batch_idx = torch.repeat_interleave(
            torch.arange(num_valid_graphs, device=device),
            num_nodes_per_graph
        )
        
        # 3. 拼接所有图的边索引和边属性
        valid_edge_indices_permuted = valid_edge_indices.permute(0, 2, 1)  # (num_valid, max_edges, 2)
        all_edge_indices_flat = valid_edge_indices_permuted[valid_edge_masks]  # (total_edges, 2)
        all_edge_indices = all_edge_indices_flat.t()  # (2, total_edges)
        
        all_edge_weights = valid_edge_weights[valid_edge_masks]  # (total_edges,)
        edge_embedding = self.edge_embedding(all_edge_weights.long())
        
        # 4. 调整边索引使其成为全局索引
        node_offsets = torch.cumsum(num_nodes_per_graph, dim=0) - num_nodes_per_graph
        num_edges_per_graph = valid_edge_masks.sum(dim=1)
        edge_index_offsets = torch.repeat_interleave(node_offsets, num_edges_per_graph)
        all_edge_indices_adjusted = all_edge_indices.long() + edge_index_offsets.unsqueeze(0)
        
        # 5. 直接创建批处理图对象
        batch_graphs = Batch(
            x=all_nodes,
            edge_index=all_edge_indices_adjusted,
            edge_attr=edge_embedding,
            batch=node_batch_idx
        )
        # ==================== 向量化图批处理结束 ====================

        # 一次性处理所有图
        gat_output = self.query_GAT(
            batch_graphs.x,
            batch_graphs.edge_index,
            batch_graphs.edge_attr
        )
        
        # 后处理
        processed_output = self.gat_ffn(gat_output)
        
        # 高效的图特征聚合
        graph_features = global_mean_pool(processed_output, batch_graphs.batch)
        
        # 高效的输出重组：使用scatter操作
        result = torch.zeros(batch_size, self.query_window_size, self.embedding_dim, device=device)
        result[valid_batch_indices, valid_query_indices] = graph_features
        
        return self.norm(result)

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(EncoderLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, x):
        x = self.self_attention_norm(x)
        x = self.attention(x, x, x)[0] + x
        x = self.ffn_norm(x)
        x = self.ffn(x) + x
        return x
        
        


class WorkloadFeatureEmbedding(nn.Module):
    def __init__(self, embedding_dim, query_window_size):
        super(WorkloadFeatureEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.query_window_size = query_window_size

        
        
        # 多头注意力机制
        self.attention = nn.ModuleList(
            [EncoderLayer(embedding_dim) for _ in range(3)]
        )
        
        # 位置编码
        # self.positional_encoding = nn.Parameter(
        #     torch.randn(query_window_size, embedding_dim)
        # )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, queries_embedding):
        """
        处理查询嵌入序列
        Args:
            queries_embedding: (batch_size, query_window_size, embedding_dim)
        """
        # 添加位置编码
        # queries_with_pos = queries_embedding + self.positional_encoding.unsqueeze(0)
        queries_with_pos = queries_embedding
        
        # 多头注意力
        for layer in self.attention:
            queries_with_pos = layer(queries_with_pos)
        
        # 残差连接和层归一化
        attn_output = self.layer_norm(queries_with_pos[:, 0, :])
        
        # 前馈网络
        ffn_output = self.ffn(attn_output)
        
        # 再次残差连接和层归一化
        # output = self.layer_norm(ffn_output + attn_output)
        
        # 聚合工作负载特征：使用平均池化
        # workload_embedding = torch.mean(output, dim=1)  # (batch_size, embedding_dim)
        
        return ffn_output


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=128):
        super(FeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # 从观察空间获取维度信息
        self.query_window_size = observation_space["wl_query_embeddings"].shape[0]
        self.max_columns_per_query = observation_space["wl_query_embeddings"].shape[1]
        self.workload_embedding_dim = observation_space["wl_query_embeddings"].shape[2]
        self.max_edges = observation_space["wl_edge_indexes"].shape[2]
        self.meta_info_dim = observation_space["meta_info_db"].shape[0]
        self.action_dim = observation_space["action"].shape[0]
        
        # 查询特征嵌入模块
        self.query_embeddings = QueryFeaturesEmbeddings(
            embedding_dim=self.workload_embedding_dim,
            query_window_size=self.query_window_size,
            max_columns_per_query=self.max_columns_per_query,
            max_edges=self.max_edges
        )
        
        # 工作负载特征嵌入模块
        self.workload_embeddings = WorkloadFeatureEmbedding(
            embedding_dim=self.workload_embedding_dim,
            query_window_size=self.query_window_size
        )
        
        # 数据库状态处理网络
        self.db_state_ffn = nn.Sequential(
            nn.Linear(self.meta_info_dim, features_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
        # 动作状态处理网络
        self.action_ffn = nn.Sequential(
            nn.Linear(self.action_dim, features_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
        # 工作负载嵌入映射到目标维度
        self.workload_projection = nn.Linear(self.workload_embedding_dim, features_dim)
        
        # 最终融合网络
        self.fusion_ffn = nn.Sequential(
            nn.Linear(features_dim * 3, features_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim * 2, features_dim),
            nn.LeakyReLU(),
            nn.Linear(features_dim, features_dim)
        )
        self.norm = nn.LayerNorm(features_dim)

    def forward(self, observation):
        """
        处理观察输入
        Args:
            observation: Dict包含以下键值对:
                - wl_query_embeddings: 查询嵌入
                - wl_query_maskings: 查询遮罩
                - wl_edge_indexes: 边索引
                - wl_edge_weights: 边权重
                - wl_edge_maskings: 边遮罩
                - meta_info_db: 数据库元信息
                - action: 动作状态
        """
        # 处理查询特征
        queries_embedding = self.query_embeddings(
            observation["wl_query_embeddings"],
            observation["wl_query_maskings"],
            observation["wl_edge_indexes"],
            observation["wl_edge_weights"],
            observation["wl_edge_maskings"]
        )
        
        # 处理工作负载特征
        workload_embedding = self.workload_embeddings(queries_embedding)
        
        # 投影工作负载嵌入到目标维度
        workload_feature = self.workload_projection(workload_embedding)
        
        # 处理数据库状态
        db_state_feature = self.db_state_ffn(observation["meta_info_db"])
        
        # 处理动作状态
        action_feature = self.action_ffn(observation["action"])
        
        # 融合所有特征
        combined_features = torch.cat([workload_feature, db_state_feature, action_feature], dim=-1)
        final_features = self.fusion_ffn(combined_features)

        final_features = self.norm(final_features)
        return final_features

