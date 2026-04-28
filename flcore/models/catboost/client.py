import json
import time
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import numpy as np
from catboost import CatBoostClassifier
from flwr.common import NDArrays, Scalar
from sklearn.model_selection import train_test_split

from flcore.metrics import calculate_metrics, find_best_threshold
from flcore.models.catboost.task import (convert_to_catboost,
                                         convert_to_model_dict)


class CatBoostClient(fl.client.NumPyClient):
    """Flower NumPyClient for fully federated CatBoost training."""

    def __init__(
        self,
        local_data: Dict,
        client_id: int,
        saving_path: str = "logs/sandbox/",
        config: Dict = None,
    ):
        self.client_id = client_id
        self.local_data = local_data
        self.config = config or {}
        self.saving_path = Path(saving_path)
        self.saving_path.mkdir(parents=True, exist_ok=True)
        (self.saving_path / "models").mkdir(exist_ok=True)

        self.model = None
        self.round_time = 0.0
        self.catboost_params = {}

        self._prepare_data()

    def _prepare_data(self):
        X_train = self.local_data["X_train"]
        y_train = self.local_data["y_train"]
        X_test = self.local_data["X_test"]
        y_test = self.local_data["y_test"]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=self.config.get("seed", 42),
            stratify=y_train,
        )

        self.local_data["X_train"] = X_train
        self.local_data["y_train"] = y_train
        self.local_data["X_val"] = X_val
        self.local_data["y_val"] = y_val
        self.local_data["X_test"] = X_test
        self.local_data["y_test"] = y_test

    def get_parameters(self, config: Dict[str, Scalar] = None) -> NDArrays:
        if self.model is None:
            return [np.array([], dtype=np.uint8)]

        model_bytes = json.dumps(convert_to_model_dict(self.model)).encode("utf-8")
        return [np.frombuffer(model_bytes, dtype=np.uint8)]

    def set_parameters(self, parameters: NDArrays):
        if len(parameters) == 0 or len(parameters[0]) == 0:
            self.model = None
            return

        model_bytes = parameters[0].tobytes()
        self.model = convert_to_catboost(model_bytes)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        server_round = int(config.get("server_round", 1))
        num_local_rounds = int(config.get("num_local_rounds", 1))
        train_method = config.get("train_method", "bagging")

        self.catboost_params = {
            k: v
            for k, v in config.items()
            if k not in ["server_round", "num_local_rounds", "train_method", "task"]
        }
        task = str(config.get("task", "binary")).lower()

        start_time = time.time()
        metrics = {}

        if server_round > 1:
            self.set_parameters(parameters)

        model = CatBoostClassifier(
            iterations=num_local_rounds,
            learning_rate=float(self.catboost_params.get("learning_rate", 0.0001)),
            depth=int(self.catboost_params.get("depth", 6)),
            random_seed=int(self.catboost_params.get("random_seed", 42)),
            loss_function=self.catboost_params.get("loss_function", "Logloss"),
            eval_metric=self.catboost_params.get("eval_metric", "AUC"),
            verbose=False,
            allow_writing_files=False,
        )

        init_model = self.model if self.model is not None else None

        model.fit(self.local_data["X_train"], self.local_data["y_train"], init_model=init_model)
        self.model = model
        model_dict = convert_to_model_dict(self.model)
        num_trees = len(model_dict.get("oblivious_trees", []))

        if server_round == 1 and task == "binary":
            local_model = CatBoostClassifier(
                iterations=num_local_rounds * int(self.config.get("num_rounds", 1)),
                learning_rate=float(self.config.get("catboost", {}).get("learning_rate", self.catboost_params.get("learning_rate", 0.1))),
                depth=int(self.config.get("catboost", {}).get("depth", self.catboost_params.get("depth", 6))),
                random_seed=int(self.config.get("seed", 42)),
                loss_function=self.catboost_params.get("loss_function", "Logloss"),
                eval_metric=self.catboost_params.get("eval_metric", "AUC"),
                verbose=False,
                allow_writing_files=False,
            )
            local_model.fit(self.local_data["X_train"], self.local_data["y_train"])

            y_val_pred = local_model.predict_proba(self.local_data["X_val"])[:, 1]
            y_val_true = self.local_data["y_val"]
            best_threshold = find_best_threshold(y_val_true, y_val_pred)

            y_test_pred = local_model.predict_proba(self.local_data["X_test"])[:, 1]
            y_test_true = self.local_data["y_test"]
            local_metrics = calculate_metrics(y_test_true, y_test_pred, threshold=best_threshold)
            metrics.update({f"local {key}": local_metrics[key] for key in local_metrics})

        model_bytes_before_extraction = json.dumps(model_dict).encode("utf-8")

        if train_method == "bagging" and num_trees > num_local_rounds:
            model_dict["oblivious_trees"] = model_dict["oblivious_trees"][num_trees - num_local_rounds : num_trees]

        model_bytes = json.dumps(model_dict).encode("utf-8")
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)

        num_examples = len(self.local_data["X_train"])
        metrics["num_examples"] = num_examples
        metrics["num_trees"] = num_trees

        local_model_path = self.saving_path / "models" / f"catboost_client_round_{server_round}.json"
        self.model.save_model(str(local_model_path), format="json")

        self.round_time = time.time() - start_time
        return [model_array], num_examples, metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        if self.model is None:
            return 0.0, 0, {}

        task = str(config.get("task", "binary")).lower()

        y_true = self.local_data["y_test"]
        y_val_true = self.local_data["y_val"]
        metrics = {}

        if task == "binary":
            y_val_pred = self.model.predict_proba(self.local_data["X_val"])[:, 1]
            best_threshold = find_best_threshold(y_val_true, y_val_pred)
            metrics_val = calculate_metrics(y_val_true, y_val_pred, threshold=best_threshold)
            metrics.update({f"val {key}": metrics_val[key] for key in metrics_val})

            y_pred = self.model.predict_proba(self.local_data["X_test"])[:, 1]
            test_metrics = calculate_metrics(y_true, y_pred, threshold=best_threshold)
            metrics.update(test_metrics)
            loss = 1 - metrics.get("auroc", 0.0)
        elif task == "regression":
            y_pred = self.model.predict(self.local_data["X_test"])
            test_metrics = calculate_metrics(y_true, y_pred, task_type="reg")
            metrics.update(test_metrics)
            loss = float(test_metrics.get("mse", 0.0))
        else:
            y_pred = self.model.predict(self.local_data["X_test"])
            accuracy = float((y_pred == y_true).mean())
            metrics["accuracy"] = accuracy
            loss = 1 - accuracy

        metrics["n samples"] = len(y_true)
        metrics["client_id"] = self.client_id
        metrics["round_time [s]"] = self.round_time

        return float(loss), len(self.local_data["X_test"]), metrics


def get_numpy(X_train, y_train, X_test, y_test) -> Dict:
    if hasattr(X_train, "values"):
        X_train = X_train.values
    if hasattr(y_train, "values"):
        y_train = y_train.values
    if hasattr(X_test, "values"):
        X_test = X_test.values
    if hasattr(y_test, "values"):
        y_test = y_test.values

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "num_examples": len(X_train),
    }


def get_client(config: Dict, data: Tuple, client_id: int) -> fl.client.Client:
    (X_train, y_train), (X_test, y_test) = data
    local_data = get_numpy(X_train, y_train, X_test, y_test)
    return CatBoostClient(
        local_data=local_data,
        client_id=client_id,
        saving_path=config.get("experiment_dir", "logs/sandbox/"),
        config=config,
    )