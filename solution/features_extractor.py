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
        
        self.edge_embedding = nn.Embedding(2, 8)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.query_GAT = GATv2Conv(
            in_channels=embedding_dim, 
            out_channels=embedding_dim, 
            heads=4, 
            edge_dim=8, 
            concat=False,
            dropout=0.1
        )
        
        self.gat_ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, query_embeddings, query_maskings, edge_indexes, edge_weights, edge_maskings):
        batch_size = query_embeddings.size(0)
        device = query_embeddings.device
        
        query_embeddings = self.layer_norm(query_embeddings)
        
        valid_nodes_mask = query_maskings.bool()  # (batch_size, query_window_size, max_columns_per_query)
        valid_edges_mask = edge_maskings.bool()   # (batch_size, query_window_size, max_edges)
        
        valid_nodes_count = valid_nodes_mask.sum(dim=-1)  # (batch_size, query_window_size)
        valid_edges_count = valid_edges_mask.sum(dim=-1)  # (batch_size, query_window_size)
        
        valid_queries_mask = (valid_nodes_count > 0) & (valid_edges_count > 0)  # (batch_size, query_window_size)
        valid_queries_indices = torch.nonzero(valid_queries_mask, as_tuple=False)  # (num_valid_queries, 2)非零坐标索引
        
        if valid_queries_indices.size(0) == 0:
            return torch.zeros(batch_size, self.query_window_size, self.embedding_dim, device=device)
        
        valid_batch_indices = valid_queries_indices[:, 0]
        valid_query_indices = valid_queries_indices[:, 1]
        
        valid_query_embeddings = query_embeddings[valid_batch_indices, valid_query_indices]  # (num_valid, max_columns, emb_dim)
        valid_node_masks = valid_nodes_mask[valid_batch_indices, valid_query_indices]         # (num_valid, max_columns)
        
        valid_edge_indices = edge_indexes[valid_batch_indices, valid_query_indices]  # (num_valid, 2, max_edges)
        valid_edge_weights = edge_weights[valid_batch_indices, valid_query_indices]  # (num_valid, max_edges)
        valid_edge_masks = valid_edges_mask[valid_batch_indices, valid_query_indices]  # (num_valid, max_edges)

        num_valid_graphs = valid_query_embeddings.size(0)

        all_nodes = valid_query_embeddings[valid_node_masks]  # (total_nodes, emb_dim)
        
        num_nodes_per_graph = valid_node_masks.sum(dim=1)  # (num_valid_graphs,)
        node_batch_idx = torch.repeat_interleave(
            torch.arange(num_valid_graphs, device=device),
            num_nodes_per_graph
        )
        
        valid_edge_indices_permuted = valid_edge_indices.permute(0, 2, 1)  # (num_valid, max_edges, 2)
        all_edge_indices_flat = valid_edge_indices_permuted[valid_edge_masks]  # (total_edges, 2)
        all_edge_indices = all_edge_indices_flat.t()  # (2, total_edges)
        
        all_edge_weights = valid_edge_weights[valid_edge_masks]  # (total_edges,)
        edge_embedding = self.edge_embedding(all_edge_weights.long())
        
        node_offsets = torch.cumsum(num_nodes_per_graph, dim=0) - num_nodes_per_graph
        num_edges_per_graph = valid_edge_masks.sum(dim=1)
        edge_index_offsets = torch.repeat_interleave(node_offsets, num_edges_per_graph)
        all_edge_indices_adjusted = all_edge_indices.long() + edge_index_offsets.unsqueeze(0)
        
        batch_graphs = Batch(
            x=all_nodes,
            edge_index=all_edge_indices_adjusted,
            edge_attr=edge_embedding,
            batch=node_batch_idx
        )

        gat_output = self.query_GAT(
            batch_graphs.x,
            batch_graphs.edge_index,
            batch_graphs.edge_attr
        )
        
        processed_output = self.gat_ffn(gat_output)
        
        graph_features = global_mean_pool(processed_output, batch_graphs.batch)
        
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

        self.attention = nn.ModuleList(
            [EncoderLayer(embedding_dim) for _ in range(3)]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, queries_embedding):
        queries_with_pos = queries_embedding
        
        for layer in self.attention:
            queries_with_pos = layer(queries_with_pos)

        attn_output = self.layer_norm(queries_with_pos[:, 0, :])
        
        ffn_output = self.ffn(attn_output)

        return ffn_output


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=128):
        super(FeaturesExtractor, self).__init__(observation_space, features_dim)
        
        self.query_window_size = observation_space["wl_query_embeddings"].shape[0]
        self.max_columns_per_query = observation_space["wl_query_embeddings"].shape[1]
        self.workload_embedding_dim = observation_space["wl_query_embeddings"].shape[2]
        self.max_edges = observation_space["wl_edge_indexes"].shape[2]
        self.meta_info_dim = observation_space["meta_info_db"].shape[0]
        self.action_dim = observation_space["action"].shape[0]
        
        self.query_embeddings = QueryFeaturesEmbeddings(
            embedding_dim=self.workload_embedding_dim,
            query_window_size=self.query_window_size,
            max_columns_per_query=self.max_columns_per_query,
            max_edges=self.max_edges
        )
        
        self.workload_embeddings = WorkloadFeatureEmbedding(
            embedding_dim=self.workload_embedding_dim,
            query_window_size=self.query_window_size
        )
        
        self.db_state_ffn = nn.Sequential(
            nn.Linear(self.meta_info_dim, features_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
        self.action_ffn = nn.Sequential(
            nn.Linear(self.action_dim, features_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
        self.workload_projection = nn.Linear(self.workload_embedding_dim, features_dim)
        
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

        queries_embedding = self.query_embeddings(
            observation["wl_query_embeddings"],
            observation["wl_query_maskings"],
            observation["wl_edge_indexes"],
            observation["wl_edge_weights"],
            observation["wl_edge_maskings"]
        )

        workload_embedding = self.workload_embeddings(queries_embedding)

        workload_feature = self.workload_projection(workload_embedding)

        db_state_feature = self.db_state_ffn(observation["meta_info_db"])

        action_feature = self.action_ffn(observation["action"])

        combined_features = torch.cat([workload_feature, db_state_feature, action_feature], dim=-1)
        final_features = self.fusion_ffn(combined_features)

        final_features = self.norm(final_features)
        return final_features

