## RLIS: A Pretrained Reinforcement Learning Framework for Cross-Database Automatic Index Selection

Automatic index selection (AIS) is crucial for optimizing database query performance, yet existing machine learning and reinforcement learning based methods suffer from significant limitations, including poor transferability across different databases, high retraining costs, and a lack of robustness to workload shifts. we propose RLIS, a framework generally applicable to cross-database scenario.

This is the implementation for the paper "A Pretrained Reinforcement Learning Framework for Cross-Database Automatic Index Selection
". The RLIS is a novel Reinforcement Learning-based pretraining framework for cross-database automatic Index Selection.



### Requirements and Installation

We recommend running this project in a virtual environment (e.g., Anaconda).

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/JiTao3/RLIS.git
    cd RLIS
    ```

2.  **Install Dependencies**
    The project dependencies are listed in the `requirements.txt` file. Please run the following command to install them:
    ```bash
    pip install -r requirements.txt
    ```
    
3. **Database Preparation**
    You need to install the corresponding database (PostgreSQL), create the database, and prepare the workload. (Training workload PRICE:  ```https://github.com/StCarmen/PRICE```)


### Project Structure

The project directory structure is organized as follows:

```
.
├── config/                             # Configuration files directory
│   ├── all_DB.json                     # Configuration of training for cross database
│   └── tpch_finetuning.json            # Example configuration for fine-tuning
├── gym_db/                             # OpenAI Gym environment definition
│   ├── envs/
│   │   └── db_env_v1.py                # Custom environment for DB index recommendation
│   └── common.py                       # Common utilities for the environment
├── selection/                          # Core index selection algorithms
│   ├── index_selection_evaluation.py   # Index evaluation logic
│   ├── cost_evaluation.py              # Query cost evaluation
│   ├── database_connector.py           # Database connection management
|   ├── ...
│   └── dbms/                           # Implementations for specific DBMS
├── solution/                           # RL solution framework
│   ├── experiment.py                   # Experiment management
│   ├── action_manager.py               # Action space manager
│   ├── observation_manager.py          # Observation space manager
│   ├── reward_calculator.py            # Reward calculator
|   ├── ...
│   └── features_extractor.py           # Custom feature extractor
├── callback.py                         # Callback for stablebaseline3
├── pt_env.py                           # Main entry point for training
├── ft_env.py                           # Entry point for fine-tuning
├── requirements.txt                    # Python dependencies
└── README.md                           # Project description
```

### Configuration

All experiment parameters are managed through JSON configuration files in the `config/` directory. A typical configuration file includes the following sections:

- **`id`**: A unique identifier for the experiment, used for logging and model checkpoint saving.
- **`workload`**: The database name paired with the corresponding SQL query workload.
  - **`benchmarks`**: Specifies the names of the database and workload to be used.
- **`rl_algorithm`**: The reinforcement learning algorithm to use (e.g., PPO), along with configurable hyperparameters.
- **`experiment_name`**: The display name of the experiment, used for organizing logs and saved models.
- **`workload_embedder`**: Hyperparameters for the workload embedding module.
- **`statistics_path`**: The file path to the cached statistics (e.g., table and column statistics) of the database.
- **...** 

### Pre-training & Fine-tuning

After configuring the experiment parameters, you can pre-train and fine-tune the model using the following commands:

```
nohup python pt_env.py ./config/all_DB.json > log/exp_pt.log  2>&1 &
nohup python ft_env.py ./config/tpch_finetuning.json > log/exp_ft.log  2>&1 &
```


