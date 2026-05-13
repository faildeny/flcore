import functools
import json
import math
import time
import warnings
from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flcore.dropout import select_clients
from flcore.metrics import metrics_aggregation_fn

warnings.filterwarnings("ignore")


def _get_tree_nums(bst_org: bytes) -> Tuple[int, int]:
    bst = json.loads(bytearray(bst_org))
    model = bst["learner"]["gradient_booster"]["model"]
    num_trees = int(model["gbtree_model_param"]["num_trees"])
    num_parallel_tree = int(model["gbtree_model_param"].get("num_parallel_tree", "1"))
    return num_trees, num_parallel_tree


def aggregate_bagging(bst_prev_org: bytes, bst_curr_org: bytes) -> bytes:
    """Bagging aggregation for XGBoost raw models."""
    if bst_prev_org == b"":
        return bst_curr_org

    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    tree_num_curr, _ = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    previous_model = bst_prev["learner"]["gradient_booster"]["model"]
    previous_model["gbtree_model_param"]["num_trees"] = str(
        tree_num_prev + tree_num_curr
    )
    iteration_indptr = previous_model["iteration_indptr"]
    previous_model["iteration_indptr"].append(iteration_indptr[-1] + tree_num_curr)

    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        previous_model["trees"].append(trees_curr[tree_count])
        previous_model["tree_info"].append(0)

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")
    return bst_prev_bytes


def get_train_method(config) -> str:
    return str(config["xgb"].get("train_method", "bagging")).lower()


def get_xgb_params(config) -> Dict[str, Scalar]:
    params: Dict[str, Scalar] = {
        "eta": float(config["xgb"].get("learning_rate")),
        "max_depth": int(config["xgb"].get("max_depth", 6)),
        "num_parallel_tree": int(config["xgb"].get("num_parallel_tree", 1)),
        "verbosity": 0,
    }

    params["objective"] = "binary:logistic"
    params["eval_metric"] = "auc"

    return params


def get_learning_rate_mode(config) -> str:
    return str(config["xgb"].get("learning_rate_mode", "uniform")).lower()


def fit_round(server_round: int, train_method: str, num_local_rounds: int, xgb_params: Dict[str, Scalar]) -> Dict[str, Scalar]:
    config = {
        "server_round": server_round,
        "train_method": train_method,
        "num_local_rounds": num_local_rounds,
    }
    config.update(xgb_params)
    return config


def evaluate_round(server_round: int, xgb_params: Dict[str, Scalar]) -> Dict[str, Scalar]:
    config = {"server_round": server_round}
    config.update(xgb_params)
    return config


class FedXgbStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        train_method: str,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        dropout_method: str = "None",
        percentage_drop: float = 0.0,
        base_learning_rate: float = 0.1,
        learning_rate_mode: str = "uniform",
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.train_method = train_method
        self.dropout_method = dropout_method
        self.percentage_drop = percentage_drop
        self.base_learning_rate = base_learning_rate
        self.learning_rate_mode = learning_rate_mode
        self.global_model = b""
        self.clients_first_round_time = {}
        self.clients_num_examples = {}
        self.time_server_round = time.time()
        self.accum_time = 0.0

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        configure_clients = super().configure_fit(server_round, parameters, client_manager)
        clients = [client for client, _ in configure_clients]
        default_fit_ins_by_client = {
            client.cid: fit_ins for client, fit_ins in configure_clients
        }
        if self.dropout_method != "None" and server_round > 1:
            clients = select_clients(
                self.dropout_method,
                self.percentage_drop,
                clients,
                self.clients_first_round_time,
                server_round,
                self.clients_num_examples,
            )

        configured_pairs: List[Tuple[ClientProxy, FitIns]] = []
        total_num_examples = sum(self.clients_num_examples.values())
        for client in clients:
            fit_ins = default_fit_ins_by_client[client.cid]
            print(f"Configuring client {client.cid} for round {server_round} with {self.clients_num_examples.get(client.cid)}")
            if self.learning_rate_mode == "weighted" and total_num_examples > 0:
                client_num_examples = self.clients_num_examples.get(client.cid)
                if client_num_examples is not None:
                    weighted_eta = float(
                        self.base_learning_rate
                        * (client_num_examples / total_num_examples)
                    )
                    fit_ins = FitIns(
                        parameters=fit_ins.parameters,
                        config={**fit_ins.config, "eta": weighted_eta},
                    )
            print(f"Client {client.cid} has learning rate: {fit_ins.config['eta']}")
            configured_pairs.append((client, fit_ins))

        return configured_pairs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        if server_round == 1:
            for client, res in results:
                self.clients_first_round_time[client.cid] = res.metrics.get(
                    "running_time", 0.0
                )
                self.clients_num_examples[client.cid] = res.num_examples

            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            else:
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            elapsed_time = time.time() - self.time_server_round
            self.accum_time += elapsed_time
            self.time_server_round = time.time()
            metrics_aggregated["training_time [s]"] = self.accum_time

            empty_parameters = fl.common.ndarrays_to_parameters(
                [np.array([], dtype=np.uint8)]
            )
            return empty_parameters, metrics_aggregated

        models = []
        for _, fit_res in results:
            ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
            if not ndarrays:
                continue
            models.append(bytes(ndarrays[0].tobytes()))

        if self.train_method == "bagging":
            for model_bytes in models:
                self.global_model = aggregate_bagging(self.global_model, model_bytes)
        else:
            largest = max(results, key=lambda res: res[1].num_examples)
            ndarrays = fl.common.parameters_to_ndarrays(largest[1].parameters)
            self.global_model = bytes(ndarrays[0].tobytes())

        parameters_aggregated = fl.common.ndarrays_to_parameters(
            [np.frombuffer(self.global_model, dtype=np.uint8)]
        )

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        else:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        elapsed_time = time.time() - self.time_server_round
        self.accum_time += elapsed_time
        self.time_server_round = time.time()
        metrics_aggregated["training_time [s]"] = self.accum_time

        return parameters_aggregated, metrics_aggregated


def get_server_and_strategy(config: Dict):
    train_method = get_train_method(config)
    learning_rate_mode = get_learning_rate_mode(config)
    num_local_rounds = 1
    xgb_params = get_xgb_params(config)

    initial_parameters = fl.common.ndarrays_to_parameters(
        [np.array([], dtype=np.uint8)]
    )

    on_fit_config_fn = functools.partial(
        fit_round,
        train_method=train_method,
        num_local_rounds=num_local_rounds,
        xgb_params=xgb_params,
    )

    on_evaluate_config_fn = functools.partial(
        evaluate_round,
        xgb_params=xgb_params,
    )

    strategy = FedXgbStrategy(
        train_method=train_method,
        min_available_clients=config["num_clients"],
        min_fit_clients=config["num_clients"],
        min_evaluate_clients=config["num_clients"],
        fit_metrics_aggregation_fn=metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=metrics_aggregation_fn,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        initial_parameters=initial_parameters,
        dropout_method=config["dropout_method"],
        percentage_drop=config["dropout"]["percentage_drop"],
        base_learning_rate=float(config["xgb"].get("learning_rate", 0.1)),
        learning_rate_mode=learning_rate_mode,
    )

    return None, strategy
