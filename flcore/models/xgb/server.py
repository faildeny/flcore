# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# XGBoost with scaled learning rate and bagging aggregation strategy
# Author: Iratxe Moya, Faildeny
# Date: January 2026
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import json
import os
from logging import WARNING, log
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from flwr.common import (EvaluateRes, FitRes, Parameters, Scalar,
                         ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from flcore.metrics import metrics_aggregation_fn

def _get_tree_nums(xgb_model_org: bytes):
    """Extract total tree numbers from XGBoost JSON model."""
    bst = json.loads(bytearray(xgb_model_org))
    model = bst["learner"]["gradient_booster"]["model"]
    tree_num = int(model["gbtree_model_param"]["num_trees"])
    paral_tree_num = int(model["gbtree_model_param"]["num_parallel_tree"])
    return tree_num, paral_tree_num

def aggregate_bagging(
    bst_prev_org: bytes,
    bst_curr_org: bytes,
) -> bytes:
    """Conduct bagging aggregation for given trees."""
    if bst_prev_org == b"":
        return bst_curr_org

    # Get the tree numbers
    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)


    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    previous_model = bst_prev["learner"]["gradient_booster"]["model"]
    previous_model["gbtree_model_param"]["num_trees"] = str(
        tree_num_prev + paral_tree_num_curr
    )
    iteration_indptr = previous_model["iteration_indptr"]
    previous_model["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        previous_model["trees"].append(trees_curr[tree_count])
        previous_model["tree_info"].append(0)


    print("Previous tree params", previous_model["gbtree_model_param"])
    # Current trees number in trees curr:
    print("Current tree params", len(bst_curr["learner"]["gradient_booster"]["model"]["trees"]))
    print("Parallel trees num: ", paral_tree_num_curr)
    # Total trees after aggregation:
    print("Total tree params", len(bst_prev["learner"]["gradient_booster"]["model"]['trees']))

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")

    return bst_prev_bytes

class FedXgbBagging(FedAvg):
    """Federated XGBoost strategy based on aggregating trees every boosting round."""

    def __init__(
        self,
        num_local_rounds: int = 5,
        xgb_params: Dict = None,
        saving_path: str = "./sandbox",
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        train_method: str = "bagging",
        fraction_train=1.0,
        fraction_evaluate=1.0,

        # --> INHERITED
        **kwargs,
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            **kwargs
        )

        self.train_method = train_method
        self.xgb_params = xgb_params or {}
        self.saving_path = Path(saving_path)
        self.saving_path.mkdir(parents=True, exist_ok=True)

        self.current_model: Optional[bytes] = b""


    def initialize_parameters(self, client_manager):
        """Start with empty model."""
        empty = np.frombuffer(b"", dtype=np.uint8)
        return ndarrays_to_parameters([empty])

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        print(f"\n[Round {server_round}] Aggregating {len(results)} clients")

        models: List[bytes] = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            model_bytes = ndarrays[0].tobytes()

            if model_bytes:
                models.append(model_bytes)

        if not models:
            return None, {}

        # -----------------------------------
        # BAGGING
        # -----------------------------------
        if self.train_method == "bagging":

            combined = self.current_model

            for m in models:
                combined = aggregate_bagging(combined, m)

        # -----------------------------------
        # CYCLIC
        # -----------------------------------
        else:
            combined = models[-1]

        self.current_model = combined

        # Save checkpoint
        self._save_checkpoint(combined, server_round)

        # Convert back to Parameters
        aggregated_params = ndarrays_to_parameters(
            [np.frombuffer(combined, dtype=np.uint8)]
        )

        # Aggregate metrics
        metrics_aggregated = {}
        total_examples = sum([fit_res.num_examples for _, fit_res in results])

        # Aggregate custom metrics if aggregation fn was provided
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        else:
            for client_proxy, fit_res in results:
                for key, value in fit_res.metrics.items():
                    # Skip non-numeric metrics (like client_id)
                    if not isinstance(value, (int, float)):
                        continue
                        
                    if key not in metrics_aggregated:
                        metrics_aggregated[key] = 0
                    # Weighted average by number of examples
                    metrics_aggregated[key] += value * fit_res.num_examples / total_examples

        print(f"[Round {server_round}] Aggregation done.")

        return aggregated_params, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        total_examples = sum(eval_res.num_examples for _, eval_res in results)

        total_loss = sum(
            eval_res.loss * eval_res.num_examples
            for _, eval_res in results
        )

        avg_loss = total_loss / total_examples

        # Aggregate metrics with weighted average
        metrics_aggregated = {}
        total_examples = sum([eval_res.num_examples for _, eval_res in results])
        
        # Aggregate custom metrics if aggregation fn was provided
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        else:
            for _, eval_res in results:
                for key, value in eval_res.metrics.items():
                    # Skip non-numeric metrics (like client_id)
                    if not isinstance(value, (int, float)):
                        continue
                        
                    if key not in metrics_aggregated:
                        metrics_aggregated[key] = 0
                    metrics_aggregated[key] += value * eval_res.num_examples / total_examples

        print(f"[Round {server_round}] Eval loss: {avg_loss:.4f}")

        return avg_loss, metrics_aggregated

    def _save_checkpoint(self, model_bytes: bytes, round_num: int):

        if not model_bytes:
            return

        checkpoint_dir = self.saving_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        bst = xgb.Booster(params=self.xgb_params)
        bst.load_model(bytearray(model_bytes))

        model_path = checkpoint_dir / f"xgboost_round_{round_num}.json"
        bst.save_model(str(model_path))

        print(f"[Checkpoint] Saved {model_path}")


def get_fit_config_fn(
    num_local_rounds: int,
    train_method: str,
    xgb_params: Dict,
) -> Callable[[int], Dict[str, Any]]:
    """Return a function that returns training configuration."""
    
    def fit_config(server_round: int) -> Dict[str, Any]:
        config = {
            "server_round": server_round,
            "num_local_rounds": num_local_rounds,
            "train_method": train_method,
        }
        # Add XGBoost parameters
        config.update(xgb_params)
        return config
    
    return fit_config

def get_evaluate_config_fn(xgb_params: Dict) -> Callable[[int], Dict[str, Any]]:
    """Return a function that returns evaluation configuration."""
    
    def evaluate_config(server_round: int) -> Dict[str, Any]:
        config = {
            "server_round": server_round,
        }
        config.update(xgb_params)
        return config
    
    return evaluate_config

def get_server_and_strategy(config: dict) -> FedXgbBagging:
    """Create strategy from config dictionary."""

    os.makedirs(config["experiment_dir"], exist_ok=True)

    task = config.get("task", "binary").lower()
    xgb_config = config.get("xgb", {})

    xgb_params = {
        "eta": xgb_config.get("learning_rate") / config.get("num_clients"),
        "max_depth": xgb_config.get("max_depth", 6),
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "num_parallel_tree": xgb_config.get("num_parallel_tree", 1),
    }

    if task == "binary":
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "auc"

    elif task == "multiclass":
        n_out = config.get("n_out")
        if n_out is None or n_out < 2:
            raise ValueError("For multiclass you must provide n_out >= 2")
        xgb_params["objective"] = "multi:softmax"
        xgb_params["eval_metric"] = "mlogloss"
        xgb_params["num_class"] = n_out

    elif task == "regression":
        xgb_params["objective"] = "reg:squarederror"
        xgb_params["eval_metric"] = "rmse"

    train_method = xgb_config.get("train_method", "bagging")  # 'bagging' or 'cyclic'
    # num_local_rounds = xgb_config.get("tree_num", 100) // config.get("num_rounds", 10)  # Trees per round
    num_local_rounds = 1
    

    print("\n" + "=" * 60)
    print("Federated XGBoost Configuration")
    print("=" * 60)
    print("Task:", task.upper())
    print("Train method:", xgb_config.get("train_method", "bagging"))
    print("Rounds:", config.get("num_rounds"))
    print("Clients:", config.get("num_clients"))
    print("XGBoost params:", xgb_params)
    print("=" * 60 + "\n")

    strategy = FedXgbBagging(
        train_method=train_method,
        num_local_rounds=num_local_rounds,
        xgb_params=xgb_params,
        saving_path=config['experiment_dir'],
        min_fit_clients=config.get('min_fit_clients', config['num_clients']),
        min_evaluate_clients=config.get('min_evaluate_clients', config['num_clients']),
        min_available_clients=config.get('min_available_clients', config['num_clients']),
        on_fit_config_fn=get_fit_config_fn(num_local_rounds, train_method, xgb_params),
        on_evaluate_config_fn=get_evaluate_config_fn(xgb_params),
        fraction_train=1.0,
        fraction_evaluate=1.0,
        fit_metrics_aggregation_fn=metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=metrics_aggregation_fn
    )

    return None, strategy
