import json
import os
from logging import WARNING, log
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from catboost import CatBoostClassifier, sum_models
from flwr.common import (EvaluateRes, FitRes, Parameters, Scalar,
                         ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from flcore.metrics import metrics_aggregation_fn
from flcore.models.catboost.task import (convert_to_catboost,
                                         convert_to_model_dict, model_to_bytes)


class FedCatBoostFullyFederated(FedAvg):
    """Fully federated CatBoost strategy with bagging/cyclic support."""

    def __init__(
        self,
        num_local_rounds: int = 1,
        catboost_params: Dict = None,
        saving_path: str = "./sandbox",
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        train_method: str = "bagging",
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            fraction_fit=fraction_train,
            fraction_evaluate=fraction_evaluate,
            **kwargs,
        )

        self.train_method = train_method
        self.num_local_rounds = num_local_rounds
        self.catboost_params = catboost_params or {}
        self.saving_path = Path(saving_path)
        self.saving_path.mkdir(parents=True, exist_ok=True)
        self.current_model: Optional[bytes] = b""

    def initialize_parameters(self, client_manager):
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

        model_blobs: List[bytes] = []
        model_num_examples: List[int] = []
        for client_proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            model_bytes = ndarrays[0].tobytes()
            if model_bytes:
                try:
                    model_dict = json.loads(model_bytes.decode("utf-8"))
                    num_trees = len(model_dict.get("oblivious_trees", []))
                except Exception:
                    num_trees = -1

                model_blobs.append(model_bytes)
                model_num_examples.append(int(fit_res.num_examples))

        if not model_blobs:
            return None, {}
        
        # ctr_merge_policy='KeepAllTables'
        # ctr_merge_policy='LeaveMostDiversifiedTable'
        ctr_merge_policy='IntersectingCountersAverage'
        # ctr_merge_policy ="FailIfCtrIntersects"

        if self.train_method == "bagging":
            models = []
            weights = []

            total_client_examples = sum(model_num_examples)
            if total_client_examples > 0:
                client_weights = [n_examples / total_client_examples for n_examples in model_num_examples]
            else:
                client_weights = [1.0 / len(model_num_examples) for _ in model_num_examples]

            if self.current_model:
                models.append(convert_to_catboost(self.current_model))
                weights.append(1.0)

            models.extend(convert_to_catboost(blob) for blob in model_blobs)
            weights.extend(client_weights)


            combined_model: CatBoostClassifier = sum_models(
                models,
                weights=weights,
                ctr_merge_policy=ctr_merge_policy,
            )
        else:
            raise ValueError(f"Unsupported train_method: {self.train_method}")
            # combined_model = convert_to_catboost(model_blobs[-1])


        # Model info handling to avoid size explosion in later rounds

        model_json = convert_to_model_dict(combined_model)
        if server_round == 1:
            self.initial_model_info = model_json.get("model_info", {})
        else:
            # Update model_info with initial model info for consistency
            model_json["model_info"] = self.initial_model_info
        
        combined_model = convert_to_catboost(json.dumps(model_json).encode("utf-8"))

        # # End of model info handling

        self.current_model = model_to_bytes(combined_model)
        
        self._save_checkpoint(combined_model, server_round)

        aggregated_params = ndarrays_to_parameters(
            [np.frombuffer(self.current_model, dtype=np.uint8)]
        )

        metrics_aggregated = {}
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        else:
            for _, fit_res in results:
                for key, value in fit_res.metrics.items():
                    if not isinstance(value, (int, float)):
                        continue
                    if key not in metrics_aggregated:
                        metrics_aggregated[key] = 0
                    metrics_aggregated[key] += value * fit_res.num_examples / total_examples

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
        total_loss = sum(eval_res.loss * eval_res.num_examples for _, eval_res in results)
        avg_loss = total_loss / total_examples

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        else:
            for _, eval_res in results:
                for key, value in eval_res.metrics.items():
                    if not isinstance(value, (int, float)):
                        continue
                    if key not in metrics_aggregated:
                        metrics_aggregated[key] = 0
                    metrics_aggregated[key] += value * eval_res.num_examples / total_examples

        return avg_loss, metrics_aggregated

    def _save_checkpoint(self, model: CatBoostClassifier, round_num: int):
        checkpoint_dir = self.saving_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        model_path = checkpoint_dir / f"catboost_round_{round_num}.json"
        model.save_model(str(model_path), format="json")


def get_fit_config_fn(
    num_local_rounds: int,
    train_method: str,
    catboost_params: Dict,
    task: str,
) -> Callable[[int], Dict[str, Any]]:
    def fit_config(server_round: int) -> Dict[str, Any]:
        config = {
            "server_round": server_round,
            "num_local_rounds": num_local_rounds,
            "train_method": train_method,
            "task": task,
        }
        config.update(catboost_params)
        return config

    return fit_config


def get_evaluate_config_fn(catboost_params: Dict, task: str) -> Callable[[int], Dict[str, Any]]:
    def evaluate_config(server_round: int) -> Dict[str, Any]:
        config = {
            "server_round": server_round,
            "task": task,
        }
        config.update(catboost_params)
        return config

    return evaluate_config


def get_server_and_strategy(config: dict):
    os.makedirs(config["experiment_dir"], exist_ok=True)

    task = config.get("task", "binary").lower()
    catboost_config = config.get("catboost", {})

    if task == "binary":
        loss_function = "Logloss"
        eval_metric = "AUC"
    elif task == "multiclass":
        loss_function = "MultiClass"
        eval_metric = "MultiClass"
    else:
        loss_function = "RMSE"
        eval_metric = "RMSE"

    catboost_params = {
        "learning_rate": float(
            catboost_config.get("learning_rate", config.get("catboost").get("learning_rate"))
        ),
        "depth": int(catboost_config.get("depth", config.get("catboost").get("depth"))),
        "random_seed": int(config.get("seed", 42)),
        "loss_function": loss_function,
        "eval_metric": eval_metric,
    }

    print(f"______________Using CatBoost parameters: {catboost_params}")

    train_method = catboost_config.get("train_method", "bagging")
    num_local_rounds = int(catboost_config.get("num_local_rounds", 1))
    

    strategy = FedCatBoostFullyFederated(
        train_method=train_method,
        num_local_rounds=num_local_rounds,
        catboost_params=catboost_params,
        saving_path=config["experiment_dir"],
        min_fit_clients=config.get("min_fit_clients", config["num_clients"]),
        min_evaluate_clients=config.get("min_evaluate_clients", config["num_clients"]),
        min_available_clients=config.get("min_available_clients", config["num_clients"]),
        on_fit_config_fn=get_fit_config_fn(num_local_rounds, train_method, catboost_params, task),
        on_evaluate_config_fn=get_evaluate_config_fn(catboost_params, task),
        fraction_train=1.0,
        fraction_evaluate=1.0,
        fit_metrics_aggregation_fn=metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=metrics_aggregation_fn,
    )

    return None, strategy