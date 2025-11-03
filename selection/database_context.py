from solution.action_manager import DenseRepresentationActionManager
from solution.observation_manager import EmbeddingObservationManager

class DatabaseContext:
    def __init__(
        self,
        database_name,
        schema,
        workloads,
        globally_indexable_columns,
        globally_indexable_columns_flat,
        statistic,
        action_storage_consumptions,
    ):
        self.database_name = database_name
        self.schema = schema
        self.workloads = workloads
        self.globally_indexable_columns = globally_indexable_columns
        self.globally_indexable_columns_flat = globally_indexable_columns_flat
        self.statistic = statistic
        self.action_storage_consumptions = action_storage_consumptions


    def get_schema(self):
        return self.schema

    def get_database_name(self):
        return self.database_name

    def switch_database(self, exp):
        exp.database_name = self.database_name
        exp.schema = self.schema
        exp.workloads = self.workloads
        exp.globally_indexable_columns = self.globally_indexable_columns
        exp.globally_indexable_columns_flat = self.globally_indexable_columns_flat
        exp.statistic = self.statistic
        exp.action_storage_consumptions = self.action_storage_consumptions
