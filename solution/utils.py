import itertools
import logging

from selection.cost_evaluation import CostEvaluation
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.index import Index
from selection.workload import Column
from torchinfo import summary
import torch

# Todo: This could be improved by passing index candidates as input
def predict_index_sizes(column_combinations, database_name):
    connector = PostgresDatabaseConnector(database_name, autocommit=True)
    connector.drop_indexes()

    cost_evaluation = CostEvaluation(connector)

    predicted_index_sizes = []

    # parent_index_size_map = {}

    for column_combination in column_combinations:
        potential_index = Index(column_combination)
        cost_evaluation.what_if.simulate_index(potential_index, True)

        full_index_size = potential_index.estimated_size
        # index_delta_size = full_index_size
        # if len(column_combination) > 1:
        #     index_delta_size -= parent_index_size_map[column_combination[:-1]]

        predicted_index_sizes.append(full_index_size)
        cost_evaluation.what_if.drop_simulated_index(potential_index)

        # parent_index_size_map[column_combination] = full_index_size

    return predicted_index_sizes


def create_column_permutation_indexes(columns, max_index_width):
    result_column_combinations = []

    table_column_dict = {}
    for column in columns:
        if column.table not in table_column_dict:
            table_column_dict[column.table] = set()
        table_column_dict[column.table].add(column)

    for length in range(1, max_index_width + 1):
        unique = set()
        count = 0
        for key, columns_per_table in table_column_dict.items():
            unique |= set(itertools.permutations(columns_per_table, length))
            count += len(set(itertools.permutations(columns_per_table, length)))
        logging.info(f"{length}-column indexes: {count}")

        result_column_combinations.append(list(unique))

    return result_column_combinations


def string_column_eq(column1, column2):
    if isinstance(column2, str):
        tmp = column2
        column2 = column1
        column1 = tmp
        return string_column_eq(column1, column2)
    return column1 == column2.table.name + "." + column2.name

def sub_index_of(index1, index2):
    index2_list = [column.table.name + "." + column.name for column in index2]
    if index1 not in index2_list:
        return False
    return True

def index_have_same_columns(index1, index2):
    index1_set = set([column.table.name + "." + column.name for column in index1])
    index2_set = set([column.table.name + "." + column.name for column in index2])
    return not index1_set.isdisjoint(index2_set)


def print_model_summary(model, experiment):
    """
    打印PPO模型的详细参数统计
    """
    print("\n" + "="*80)
    print("PPO模型结构和参数统计")
    print("="*80)
    
    # 获取观察空间的示例输入
    obs_space = model.observation_space
    
    print(f"观察空间维度:")
    for key, space in obs_space.spaces.items():
        print(f"  {key}: {space.shape}")
    
    print(f"\n动作空间维度: {model.action_space.shape}")
    
    # 打印特征提取器详细信息
    print(f"\n{'='*50}")
    print("1. 特征提取器 (FeaturesExtractor)")
    print(f"{'='*50}")
    
    try:
        features_extractor = model.policy.features_extractor
        print(f"特征提取器输出维度: {features_extractor.features_dim}")
        
        # 创建示例观察输入 - 改进版本
        device = next(features_extractor.parameters()).device
        sample_obs = {}
        
        for key, space in obs_space.spaces.items():
            if key == "wl_query_embeddings":
                sample_obs[key] = torch.randn(1, *space.shape, dtype=torch.float32, device=device)
            elif key == "wl_query_maskings":
                sample_obs[key] = torch.randint(0, 2, (1, *space.shape), dtype=torch.bool, device=device)
            elif key == "wl_edge_indexes":
                # 确保边索引在有效范围内
                max_nodes = space.shape[1] if len(space.shape) > 1 else 10
                sample_obs[key] = torch.randint(0, max_nodes, (1, *space.shape), dtype=torch.long, device=device)
            elif key == "wl_edge_weights":
                sample_obs[key] = torch.randint(0, 2, (1, *space.shape), dtype=torch.long, device=device)
            elif key == "wl_edge_maskings":
                sample_obs[key] = torch.randint(0, 2, (1, *space.shape), dtype=torch.bool, device=device)
            elif key == "meta_info_db":
                sample_obs[key] = torch.randn(1, *space.shape, dtype=torch.float32, device=device)
            elif key == "action":
                sample_obs[key] = torch.randn(1, *space.shape, dtype=torch.float32, device=device)
        
        # 尝试前向传播测试
        try:
            with torch.no_grad():
                features_extractor.eval()
                output = features_extractor(sample_obs)
                print(f"✓ 前向传播测试成功，输出形状: {output.shape}")
        except Exception as forward_e:
            print(f"✗ 前向传播测试失败: {forward_e}")
        
        # 详细的手动参数统计
        print(f"\n特征提取器子模块分析:")
        total_params = 0
        trainable_params = 0
        
        for name, module in features_extractor.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += module_params
            trainable_params += module_trainable
            print(f"  {name}: {module_params:,} 参数 ({module_trainable:,} 可训练)")
            
            # 递归显示子模块
            for sub_name, sub_module in module.named_children():
                sub_params = sum(p.numel() for p in sub_module.parameters())
                if sub_params > 0:
                    print(f"    └─ {sub_name}: {sub_params:,} 参数")
        
        print(f"\n特征提取器总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
    except Exception as e:
        print(f"特征提取器分析失败: {e}")
    
    # *** 新增：打印MLP提取器信息 (这是配置文件中net_arch实际控制的部分) ***
    print(f"\n{'='*50}")
    print("1.5. MLP提取器 (核心网络层 - net_arch配置)")
    print(f"{'='*50}")
    
    try:
        mlp_extractor = model.policy.mlp_extractor
        print(f"策略网络 (Policy Network) MLP结构:")
        print(mlp_extractor.policy_net)
        
        print(f"\n价值网络 (Value Network) MLP结构:")
        print(mlp_extractor.value_net)
        
        # 显示配置信息与实际结构的对应关系
        config = experiment.config
        net_arch = config.get('rl_algorithm', {}).get('model_params', {}).get('model_architecture', {}).get('net_arch', {})
        print(f"\n配置文件中的net_arch: {net_arch}")
        
        mlp_params = sum(p.numel() for p in mlp_extractor.parameters())
        mlp_trainable = sum(p.numel() for p in mlp_extractor.parameters() if p.requires_grad)
        print(f"MLP提取器参数数量: {mlp_params:,} ({mlp_trainable:,} 可训练)")
        
        # 分别统计策略和价值网络的参数
        policy_mlp_params = sum(p.numel() for p in mlp_extractor.policy_net.parameters())
        value_mlp_params = sum(p.numel() for p in mlp_extractor.value_net.parameters())
        print(f"  策略MLP参数: {policy_mlp_params:,}")
        print(f"  价值MLP参数: {value_mlp_params:,}")
        
    except Exception as e:
        print(f"无法获取MLP提取器信息: {e}")
    
    # 打印策略网络 (Actor) 输出层信息
    print(f"\n{'='*50}")
    print("2. 策略网络输出层 (Actor Output Layer)")
    print(f"{'='*50}")
    
    try:
        actor_net = model.policy.action_net
        print(f"Actor输出层结构:")
        print(actor_net)
        print(f"注意：这只是输出层，完整的策略网络包括上面的MLP层")
        
        actor_params = sum(p.numel() for p in actor_net.parameters())
        actor_trainable = sum(p.numel() for p in actor_net.parameters() if p.requires_grad)
        print(f"Actor输出层参数数量: {actor_params:,} ({actor_trainable:,} 可训练)")
        
    except Exception as e:
        print(f"无法获取Actor网络信息: {e}")
    
    # 打印价值网络 (Critic) 输出层信息
    print(f"\n{'='*50}")
    print("3. 价值网络输出层 (Value Output Layer)")
    print(f"{'='*50}")
    
    try:
        value_net = model.policy.value_net
        print(f"Value输出层结构:")
        print(value_net)
        print(f"注意：这只是输出层，完整的价值网络包括上面的MLP层")
        
        value_params = sum(p.numel() for p in value_net.parameters())
        value_trainable = sum(p.numel() for p in value_net.parameters() if p.requires_grad)
        print(f"Value输出层参数数量: {value_params:,} ({value_trainable:,} 可训练)")
        
    except Exception as e:
        print(f"无法获取Value网络信息: {e}")
    
    # 打印整体模型统计
    print(f"\n{'='*50}")
    print("4. 整体模型统计")
    print(f"{'='*50}")
    
    try:
        total_params = sum(p.numel() for p in model.policy.parameters())
        trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        
        print(f"模型总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"模型内存占用 (估算): {total_params * 4 / 1024 / 1024:.2f} MB")  # 假设float32
        
        # 详细分组统计
        print(f"\n详细参数分布:")
        features_params = sum(p.numel() for p in model.policy.features_extractor.parameters())
        mlp_params = sum(p.numel() for p in model.policy.mlp_extractor.parameters()) if hasattr(model.policy, 'mlp_extractor') else 0
        actor_params = sum(p.numel() for p in model.policy.action_net.parameters())
        value_params = sum(p.numel() for p in model.policy.value_net.parameters())
        
        print(f"  特征提取器: {features_params:,} ({features_params/total_params*100:.1f}%)")
        print(f"  MLP提取器: {mlp_params:,} ({mlp_params/total_params*100:.1f}%)")
        print(f"  策略输出层: {actor_params:,} ({actor_params/total_params*100:.1f}%)")
        print(f"  价值输出层: {value_params:,} ({value_params/total_params*100:.1f}%)")
        
    except Exception as e:
        print(f"无法获取整体模型统计: {e}")
    
    # 打印配置信息
    print(f"\n{'='*50}")
    print("5. 模型配置信息")
    print(f"{'='*50}")
    
    try:
        config = experiment.config
        rl_config = config.get('rl_algorithm', {})
        model_params = rl_config.get('model_params', {})
        model_args = model_params.get('args', {})
        model_arch = model_params.get('model_architecture', {})
        
        print(f"算法: {rl_config.get('algorithm', 'N/A')}")
        print(f"策略类型: {rl_config.get('policy', 'N/A')}")
        print(f"学习率: {model_params.get('learning_rate', 'N/A')}")
        print(f"批次大小: {model_args.get('batch_size', 'N/A')}")
        print(f"折扣因子: {model_args.get('gamma', 'N/A')}")
        print(f"网络架构: {model_arch.get('net_arch', 'N/A')}")
        print(f"特征维度: {model_arch.get('features_extractor_kwargs', {}).get('features_dim', 'N/A')}")
        
    except Exception as e:
        print(f"无法获取配置信息: {e}")
    
    print("="*80)
