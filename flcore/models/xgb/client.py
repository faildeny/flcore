import json
import time
from typing import Dict, Tuple

import flwr as fl
import numpy as np
import xgboost as xgb
from flwr.common import NDArrays, Scalar
from sklearn.model_selection import train_test_split

from flcore.metrics import calculate_metrics, find_best_threshold


def _strip_sum_hessian(model_bytes: bytes) -> bytes:
    """
    Remove sum_hessian values before sending the model to prevent potential security breach 
    found in Timberstrike study https://arxiv.org/abs/2506.07605
    """
    model = json.loads(model_bytes.decode("utf-8"))

    def _zero_sum_hessian(obj):
        if isinstance(obj, dict):
            if "sum_hessian" in obj:
                value = obj["sum_hessian"]
                if isinstance(value, list):
                    obj["sum_hessian"] = [0.0 for _ in value]
                elif isinstance(value, (int, float)):
                    obj["sum_hessian"] = 0.0
                else:
                    obj["sum_hessian"] = 0.0
            for value in obj.values():
                _zero_sum_hessian(value)
        elif isinstance(obj, list):
            for item in obj:
                _zero_sum_hessian(item)

    _zero_sum_hessian(model)

    return json.dumps(model).encode("utf-8")


class XGBoostClient(fl.client.NumPyClient):
    """Client for federated XGBoost training based on FedXGBagging.
    Each clients trains for one boosting round and sends only the newly trained trees to the server. 
    The server aggregates by simply concatenating the trees (bagging) in each boosting round.

    """
    
    def __init__(
        self,
        local_data: Dict,
        client_id: int,
        config: Dict = None
    ):
        """
        Initialize XGBoost client.
        """
        self.client_id = client_id
        self.local_data = local_data
        self.config = config
        
        # Local model
        self.bst = None
        self.xgb_params = {}
        self.dtrain = None
        self.dtest = None
        self.label_encoder = None  # For categorical target encoding

        self.round_time = None
        self.fairness_attribute = config.get("parititon_by_attribute", None)
        if self.fairness_attribute is None:
            self.fairness_attribute = config.get("partition_by_attribute", None)
        self.fairness_attributes = (
            [self.fairness_attribute] if self.fairness_attribute is not None else None
        )
        
        # Prepare data
        self._prepare_data()

    def _get_fairness_kwargs(self, X_subset):
        if self.fairness_attributes is None:
            return {}
        return {"X": X_subset, "fairness_attributes": self.fairness_attributes}
    
    def _prepare_data(self):
        """Prepare train/val split and convert to DMatrix format."""

        X_train = self.local_data['X_train']
        y_train = self.local_data['y_train']
        X_test = self.local_data['X_test']
        y_test = self.local_data['y_test']

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=self.config['seed'], stratify=y_train)
    
        self.local_data['X_train'] = X_train
        self.local_data['y_train'] = y_train
        self.local_data['X_val'] = X_val
        self.local_data['y_val'] = y_val

        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dtest = xgb.DMatrix(X_test, label=y_test)
        self.dval = xgb.DMatrix(X_val, label=y_val)

    
    def get_parameters(self, config: Dict[str, Scalar] = None) -> NDArrays:
        """Get current model parameters."""
        if self.bst is None:
            return [np.array([], dtype=np.uint8)]
        
        model_bytes = _strip_sum_hessian(self.bst.save_raw("json"))
        return [np.frombuffer(model_bytes, dtype=np.uint8)]
    
    def set_parameters(self, parameters: NDArrays):
        """Set model parameters."""
        if len(parameters) == 0 or len(parameters[0]) == 0:
            self.bst = None
            return
        
        self.bst = xgb.Booster(params=self.xgb_params)
        self.bst.load_model(bytearray(parameters[0].tobytes()))
        
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model for one boosting round."""
        
        # Extract config
        server_round = int(config.get("server_round", 1))

        # Update XGBoost parameters from config
        self.xgb_params = {
            k: v for k, v in config.items()
            if k not in ["server_round", "num_local_rounds", "train_method"]
        }

        start_time = time.time()
        # Prepare metrics
        metrics = {}
        num_examples = len(self.local_data['X_train'])

        if server_round == 1:
            # Warm-up round: only compute local metrics, and send client size info for eta calculation,
            # do not train the federated model.
            local_xgb_params = self.xgb_params.copy()
            local_bst = xgb.train(
                local_xgb_params,
                self.dtrain,
                num_boost_round=self.config['num_rounds'],
            )
            # Get validation score, find best threshold and calculate test metrics
            y_val_pred = local_bst.predict(self.dval)
            y_val_true = self.local_data['y_val']
            best_threshold = find_best_threshold(y_val_true, y_val_pred)
            # Get test metrics and add to metrics with 'local' prefix
            y_test_pred = local_bst.predict(self.dtest)
            y_test_true = self.local_data['y_test']
            local_metrics = calculate_metrics(
                y_test_true,
                y_test_pred,
                threshold=best_threshold,
                **self._get_fairness_kwargs(self.local_data['X_test']),
            )
            metrics.update({f"local {key}": local_metrics[key] for key in local_metrics})

            model_array = np.array([], dtype=np.uint8)
            metrics['num_trees'] = 0
            self.bst = None
        else:
            if self.bst is None:
                # Train from scratch
                self.bst = xgb.train(
                    self.xgb_params,
                    self.dtrain,
                    num_boost_round=1,
                )
            else:
                self.set_parameters(parameters)
                self.bst.update(self.dtrain, self.bst.num_boosted_rounds())
            # Extract the new trained trees to send to server
            num_trees = self.bst.num_boosted_rounds()
            if num_trees > 1:
                # Extract last 
                model_to_send = self.bst[num_trees - 1 : num_trees]
            else:
                # First round
                model_to_send = self.bst

            # Serialize model
            model_bytes = _strip_sum_hessian(model_to_send.save_raw("json"))
            model_array = np.frombuffer(model_bytes, dtype=np.uint8)
            metrics['num_trees'] = self.bst.num_boosted_rounds()

        metrics['num_examples'] = num_examples
        metrics["client_id"] = self.client_id

        self.round_time = (time.time() - start_time)

        return [model_array], num_examples, metrics
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate on local test data."""
        
        # Update XGBoost parameters
        self.xgb_params = {
            k: v for k, v in config.items()
            if k not in ["server_round"]
        }
        
        # Load global model
        self.set_parameters(parameters)
        
        if self.bst is None:
            return 0.0, -1, {}
        
        metrics = {}
        
        # Get valiidation prediction for threshold finding and additional metrics
        y_val_pred = self.bst.predict(self.dval)
        y_val_true = self.local_data['y_val']
        best_threshold = find_best_threshold(y_val_true, y_val_pred)
        metrics_val = calculate_metrics(
            y_val_true,
            y_val_pred,
            threshold=best_threshold,
            **self._get_fairness_kwargs(self.local_data['X_val']),
        )
        metrics.update({f"val {key}": metrics_val[key] for key in metrics_val})

        y_pred = self.bst.predict(self.dtest)
        y_true = self.local_data['y_test']
        
        general_metrics = calculate_metrics(
            y_true,
            y_pred,
            threshold=best_threshold,
            **self._get_fairness_kwargs(self.local_data['X_test']),
        )
        metrics.update(general_metrics)
        # Add n samples to metrics
        metrics['n samples'] = len(y_true)
        metrics['client_id'] = self.client_id
        metrics['round_time [s]'] = self.round_time
        primary_metric = metrics.get('auroc', 0)
        loss = 1 - primary_metric
        
        num_examples = len(self.local_data['X_test'])
        
        return loss, num_examples, metrics


def get_client(config: Dict, data: Tuple, client_id: int) -> fl.client.Client:
    """Create XGBoost client."""
    
    (X_train, y_train), (X_test, y_test) = data
    
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'num_examples': len(X_train),
    }
    
    client = XGBoostClient(
        local_data=data,
        client_id=client_id,
        config=config
    )
    
    return client
