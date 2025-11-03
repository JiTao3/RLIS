import numpy as np
import logging
from solution.statistics import Statistics
from selection.workload import Workload


class SQLWorkloadEmbedder:
    def __init__(
        self,
        workload_window_size: int,
        max_columns_per_query: int,
        max_edges: int,
        sql_predicates_vec_size: int,
        sql_embedding_dim: int,
    ):
        self.workload_window_size = workload_window_size
        self.max_columns_per_query = max_columns_per_query
        self.max_edges = max_edges
        self.sql_predicates_vec_size = sql_predicates_vec_size
        self.embedding_dim = sql_embedding_dim

    def get_embeddings(self, workload: Workload, statistic: Statistics):
        """
        a window of workload
        """
        embeddings = []
        query_maskings = []
        edge_indexes = []
        edge_weights = []
        edge_maskings = []
        logging.info(" workload embedding start")
        for query in workload.queries:
            query_vec, query_masking, edge_index, edge_weight, edge_masking = query.to_vector(
                statistic, self.max_columns_per_query, self.sql_predicates_vec_size, self.max_edges
            )
            embeddings.append(query_vec)
            query_maskings.append(query_masking)
            edge_indexes.append(edge_index)
            edge_weights.append(edge_weight)
            edge_maskings.append(edge_masking)
        logging.info(" workload embedding end")
        return {
            "embeddings": np.array(embeddings, dtype=np.float32),
            "query_maskings": np.array(query_maskings, dtype=np.int32),
            "edge_indexes": np.array(edge_indexes, dtype=np.int32),
            "edge_weights": np.array(edge_weights, dtype=np.int32),
            "edge_maskings": np.array(edge_maskings, dtype=np.int32),
        }

    def get_embeddings_batch(self, workloads: list[Workload], statistic: Statistics):
        """
        a batch of workload
        """
        embeddings = []
        query_maskings = []
        edge_indexes = []
        edge_weights = []
        edge_maskings = []
        for workload in workloads:
            w_embedding = self.get_embeddings(workload, statistic)
            embeddings.append(w_embedding["embeddings"])
            query_maskings.append(w_embedding["query_maskings"])
            edge_indexes.append(w_embedding["edge_indexes"])
            edge_weights.append(w_embedding["edge_weights"])
            edge_maskings.append(w_embedding["edge_maskings"])
        return {
            "embeddings": np.array(embeddings),
            "query_maskings": np.array(query_maskings),
            "edge_indexes": np.array(edge_indexes),
            "edge_weights": np.array(edge_weights),
            "edge_maskings": np.array(edge_maskings),
        }
