import os
import subprocess
import sys
import time
from itertools import product

import yaml

data_normalization = ["global"]
n_features = [None]

# Number of Clients ablation experiment
experiment_name = "num_clients_ablation"
benchmark_dir = "benchmark_results"
model_names = [
    "logistic_regression",
    "elastic_net",
    "lsvc",
    "random_forest",
    "balanced_random_forest",
    "xgb"
    ]
datasets = ["diabetes"]
num_clients = [1,3,5,10,20,50]
dirichlet_alpha = [0.7, None]

# # General benchmark experiment
# experiment_name = "general"
# benchmark_dir = "benchmark_results_general_local_fixed"
# model_names = [
#    "logistic_regression",
# #    "elastic_net",
# #    "lsvc",
#     # "random_forest",
#     "balanced_random_forest",
#     "xgb"
#     ]
# datasets = ["kaggle_hf", "diabetes", "ukbb_cvd"]
# # datasets = ["ukbb_cvd"]
# num_clients = [10]
# dirichlet_alpha = [0.7]
# data_normalization = ["global"]
# n_features = [None]

# Fairness benchmark experiment
# experiment_name = "fairness"
# benchmark_dir = "benchmark_results_fairness_10_clients"
# model_names = [
# #    "logistic_regression",
# #    "elastic_net",
# #    "lsvc",
#     # "random_forest",
#     # "balanced_random_forest",
#     "xgb"
#     ]
# datasets = ["diabetes"]
# num_clients = [10]
# dirichlet_alpha = [0.7, None]
data_normalization = ["global"]
n_features = [None]

os.makedirs(benchmark_dir, exist_ok=True)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


config_path = os.path.join(benchmark_dir, "config.yaml")
log_file_path = os.path.join(benchmark_dir, "run_log.txt")

with open(config_path, "w") as f:
    yaml.dump(config, f)

config['data_path'] = 'dataset/'
config['experiment']['log_path'] = benchmark_dir

if "fairness" in experiment_name:
    config['parititon_by_attribute'] = "Sex"
else:
    config['parititon_by_attribute'] = None

start_time = time.time()

# Flatten the nested loops into a single iterator
parameters = product(datasets, num_clients, dirichlet_alpha, model_names, data_normalization, n_features)

try:
    for ds_name, n_client, alpha, m_name, norm, n_feat in parameters:
        print(f"Running benchmark: {ds_name}, {m_name}, clients: {n_client}, alpha: {alpha}, normalization: {norm}, features: {n_feat}")
        
        if "kaggle_hf" in ds_name: n_client = 4
        if "ukbb_cvd" in ds_name: n_client = 20
        # if "diabetes" in ds_name: n_client = 10
        if "forest" in m_name:
            config['num_rounds'] = 1
        elif "xgb" in m_name:
            config['num_rounds'] = 60
        else:
            config['num_rounds'] = 60
        # Update config dictionary
        config.update({
            'model': m_name,
            'dataset': ds_name,
            'num_clients': n_client,
            'dirichlet_alpha': alpha,
            'data_normalization': norm,
            'n_features': n_feat
        })
        
          # Set number of jobs for parallel processing

        config['experiment']['name'] = f"{experiment_name}_{ds_name}_{m_name}_c{n_client}_a{alpha}_norm{norm}_feat{n_feat}"

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # subprocess.run is cleaner for synchronous execution
        # Use a list for the command to avoid shell=True security/cleanup issues
        cmd = f"python repeated.py {config_path} | tee {log_file_path}"
        subprocess.run(cmd, shell=True, check=True)

except KeyboardInterrupt:
    print("\nBenchmark interrupted by user. Exiting...")
    sys.exit(1)


total_time = time.time() - start_time
print("Benchmark experiments finished in", total_time/60, " minutes")
