# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# XGBoost
# Author: Iratxe Moya
# Date: January 2026
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import os
import time
from typing import Dict, Tuple, List
import flwr as fl
from flwr.common import NDArrays, Scalar
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from pathlib import Path
from flcore.metrics import calculate_metrics, find_best_threshold


class XGBoostClient(fl.client.NumPyClient):
    """Flower client for federated XGBoost training.
    
    Supports two training methods:
    - bagging: Each client trains new trees, server combines all trees
    - cyclic: Each client refines the global model sequentially
    """
    
    def __init__(
        self,
        local_data: Dict,
        client_id: int,
        saving_path: str = "logs/sandbox/",
        config: Dict = None
    ):
        """
        Initialize XGBoost client.
        
        Args:
            local_data: Dictionary containing:
                - X_train: Training features
                - y_train: Training labels
                - X_test: Test features
                - y_test: Test labels
            saving_path: Path to save local models and logs
        """
        self.client_id = client_id
        self.local_data = local_data
        self.saving_path = Path(saving_path)
        self.saving_path.mkdir(parents=True, exist_ok=True)
        self.config = config
        # Create models directory
        models_dir = self.saving_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Local model
        self.bst = None
        self.xgb_params = {}
        self.dtrain = None
        self.dtest = None
        self.label_encoder = None  # For categorical target encoding

        self.round_time = None
        
        # Prepare data
        self._prepare_data()
        
        print(f"[Client] Initialized")
        print(f"[Client] Training samples: {len(self.local_data['X_train'])}")
        print(f"[Client] Test samples: {len(self.local_data['X_test'])}")
    
    def _prepare_data(self):
        """Convert data to DMatrix format for XGBoost."""

        X_train = self.local_data['X_train']
        y_train = self.local_data['y_train']
        X_test = self.local_data['X_test']
        y_test = self.local_data['y_test']

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=self.config['seed'], stratify=y_train)
    
        self.local_data['X_train'] = X_train
        self.local_data['y_train'] = y_train
        self.local_data['X_val'] = X_val
        self.local_data['y_val'] = y_val

        # Handle categorical labels (for multiclass classification)
        # XGBoost requires numeric labels, not strings
        if hasattr(y_train, 'dtype') and y_train.dtype == 'object':
            print(f"[Client] Detected categorical labels, encoding...")
            from sklearn.preprocessing import LabelEncoder
            
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.label_encoder.transform(y_test)
            y_val = self.label_encoder.transform(y_val)
            
            # Update local_data with encoded labels
            self.local_data['y_train'] = y_train
            self.local_data['y_test'] = y_test
            self.local_data['y_val'] = y_val
            
            print(f"[Client] Label mapping: {dict(enumerate(self.label_encoder.classes_))}")
            print(f"[Client] Encoded labels - Train: {np.unique(y_train)}, Test: {np.unique(y_test)}")
        else:
            self.label_encoder = None
        
        # Create DMatrix objects
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dtest = xgb.DMatrix(X_test, label=y_test)
        self.dval = xgb.DMatrix(X_val, label=y_val)
        
        print(f"[Client] Data prepared as DMatrix")
    
    def get_parameters(self, config: Dict[str, Scalar] = None) -> NDArrays:
        """Return current model parameters."""
        if self.bst is None:
            # Return empty parameters if no model yet
            return [np.array([], dtype=np.uint8)]
        
        # Serialize model
        model_bytes = self.bst.save_raw("json")
        return [np.frombuffer(model_bytes, dtype=np.uint8)]
    
    def set_parameters(self, parameters: NDArrays):
        """Set model parameters from server."""
        if len(parameters) == 0 or len(parameters[0]) == 0:
            # No parameters to load (first round)
            self.bst = None
            return
        
        # Load model from bytes
        model_bytes = bytearray(parameters[0].tobytes())
        self.bst = xgb.Booster(params=self.xgb_params)
        self.bst.load_model(model_bytes)
        
        print(f"[Client] Loaded global model with {self.bst.num_boosted_rounds()} trees")
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Training configuration from server
        
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        
        # Extract config
        server_round = int(config.get("server_round", 1))
        num_local_rounds = int(config.get("num_local_rounds", 5))
        train_method = config.get("train_method", "bagging")

        # Update XGBoost parameters from config
        self.xgb_params = {
            k: v for k, v in config.items()
            if k not in ["server_round", "num_local_rounds", "train_method"]
        }
        print(f"\n[Client] === Round {server_round} - FIT ===")
        print(f"[Client] Method: {train_method}")
        print(f"[Client] Local rounds: {num_local_rounds}")
        start_time = time.time()
        # Prepare metrics
        metrics = {}
        if server_round == 1:
            # First round: train from scratch
            print(f"[Client] Training from scratch...")
            self.bst = xgb.train(
                self.xgb_params,
                self.dtrain,
                num_boost_round=num_local_rounds,
            )
            # Train the model for total num_local_rounds to get a local training score
            local_xgb_params = self.xgb_params.copy()
            # Modify learning rate to 0.2 for local training
            local_xgb_params['eta'] = self.config['xgb']['learning_rate']
            local_bst = xgb.train(
                local_xgb_params,
                self.dtrain,
                num_boost_round=num_local_rounds*self.config.get("num_rounds", 1),
            )
            # Get validation score, find best threshold and calculate test metrics
            y_val_pred = local_bst.predict(self.dval)
            y_val_true = self.local_data['y_val']
            best_threshold = find_best_threshold(y_val_true, y_val_pred)
            # Get test metrics and add to metrics with 'local' prefix
            y_test_pred = local_bst.predict(self.dtest)
            y_test_true = self.local_data['y_test']
            local_metrics = calculate_metrics(y_test_true, y_test_pred, threshold=best_threshold)
            metrics.update({f"local {key}": local_metrics[key] for key in local_metrics})
        else:
            # Subsequent rounds: load global model and continue training
            self.set_parameters(parameters)
            
            if self.bst is None:
                # Fallback: train from scratch if loading failed
                print(f"[Client] Warning: Could not load model, training from scratch")
                self.bst = xgb.train(
                    self.xgb_params,
                    self.dtrain,
                    num_boost_round=num_local_rounds,
                )
            else:
                # Continue training
                print(f"[Client] Continuing training from global model...")
                initial_trees = self.bst.num_boosted_rounds()
                
                # Update trees based on local training data
                for i in range(num_local_rounds):
                    self.bst.update(self.dtrain, self.bst.num_boosted_rounds())
                
                final_trees = self.bst.num_boosted_rounds()
                print(f"[Client] Trained {final_trees - initial_trees} new trees (total: {final_trees})")
        
        print(f"[Client] Trained {self.bst.num_boosted_rounds()} boosting rounds with num parallel trees: {self.xgb_params.get('num_parallel_tree', 1)}")
        
        # For bagging: return only the last N trees
        # For cyclic: return the entire model
        if train_method == "bagging":
            # Extract only the newly trained trees
            num_trees = self.bst.num_boosted_rounds()
            if num_trees > num_local_rounds:
                # Slice to get last num_local_rounds trees
                model_to_send = self.bst[num_trees - num_local_rounds : num_trees]
                print(f"[Client] Sending last {num_local_rounds} trees (bagging mode)")
            else:
                model_to_send = self.bst
                print(f"[Client] Sending all {num_trees} trees")
        else:
            # Cyclic: send entire model
            model_to_send = self.bst
            print(f"[Client] Sending entire model (cyclic mode)")
        
        # Serialize model
        model_bytes = model_to_send.save_raw("json")
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        
        # Get number of training examples
        num_examples = len(self.local_data['X_train'])
        
        metrics['num_examples'] = num_examples
        metrics['num_trees'] = self.bst.num_boosted_rounds()
        
        
        # Save local model
        local_model_path = self.saving_path / "models" / f"xgboost_client__round_{server_round}.json"
        self.bst.save_model(str(local_model_path))
        print(f"[Client] Saved local model to {local_model_path}")

        self.round_time = (time.time() - start_time)
        
        return [model_array], num_examples, metrics
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the global model on local test data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration from server
        
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        
        server_round = int(config.get("server_round", 0))
        
        print(f"\n[Client] === Round {server_round} - EVALUATE ===")
        
        # Update XGBoost parameters
        self.xgb_params = {
            k: v for k, v in config.items()
            if k not in ["server_round"]
        }
        
        # Load global model
        self.set_parameters(parameters)
        
        if self.bst is None:
            print(f"[Client] Warning: No model to evaluate")
            return 0.0, 0, {}
        
        # Evaluate on test set
        eval_results = self.bst.eval_set(
            evals=[(self.dtest, "test")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        
        print(f"[Client] Evaluation results: {eval_results}")
        
        # Parse evaluation results
        # Format: "[0]\ttest-auc:0.85123"
        metrics = {}
        try:
            parts = eval_results.split("\t")
            for part in parts[1:]:  # Skip the iteration number
                metric_name, metric_value = part.split(":")
                metric_name = metric_name.replace("test-", "")
                metrics[metric_name] = float(metric_value)
        except Exception as e:
            print(f"[Client] Warning: Could not parse metrics: {e}")
        
        # Get valiidation prediction for threshold finding and additional metrics
        y_val_pred = self.bst.predict(self.dval)
        y_val_true = self.local_data['y_val']
        best_threshold = find_best_threshold(y_val_true, y_val_pred)
        metrics_val = calculate_metrics(y_val_true, y_val_pred, threshold=best_threshold)
        metrics.update({f"val {key}": metrics_val[key] for key in metrics_val})

        
        # Get predictions for additional metrics
        y_pred = self.bst.predict(self.dtest)
        y_true = self.local_data['y_test']
        
        # Determine task type from objective
        objective = self.xgb_params.get("objective", "")
        
        # Calculate additional metrics based on task type
        if objective.startswith("binary"):
            # Binary classification
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            general_metrics = calculate_metrics(y_true, y_pred, threshold=best_threshold)
            metrics.update(general_metrics)
            # Add n samples to metrics
            metrics['n samples'] = len(y_true)
            metrics['client_id'] = self.client_id
            metrics['round_time [s]'] = self.round_time
            # Loss is 1 - AUC for binary
            primary_metric = metrics.get('auc', 0)
            loss = 1 - primary_metric
            
        elif objective.startswith("multi"):
            # Multiclass classification
            from sklearn.metrics import accuracy_score, f1_score
            
            # y_pred is already the predicted class (not probabilities)
            y_pred_class = y_pred.astype(int)
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred_class))
            metrics['f1_macro'] = float(f1_score(y_true, y_pred_class, average='macro', zero_division=0))
            metrics['f1_weighted'] = float(f1_score(y_true, y_pred_class, average='weighted', zero_division=0))
            
            # Loss is mlogloss (already calculated by XGBoost)
            loss = metrics.get('mlogloss', 1.0)
            
        elif objective.startswith("reg"):
            # Regression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
            
            # Loss is RMSE (primary metric for regression)
            loss = metrics.get('rmse', metrics['mse'] ** 0.5)
        else:
            # Unknown task, use default loss
            loss = 1.0
        
        num_examples = len(self.local_data['X_test'])
        
        print(f"[Client] Metrics: {metrics}")
        print(f"[Client] Loss: {loss:.4f}")
        
        return loss, num_examples, metrics


def get_numpy(X_train, y_train, X_test, y_test, time_col=None, event_col=None) -> Dict:
    """Convert data to dictionary format expected by client.
    
    Args:
        X_train: Training features (numpy array or pandas DataFrame)
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        time_col: Optional time column for survival analysis
        event_col: Optional event column for survival analysis
    
    Returns:
        Dictionary with X_train, y_train, X_test, y_test
    """
    
    # Convert to numpy if needed
    if hasattr(X_train, 'values'):  # pandas DataFrame
        X_train = X_train.values
    if hasattr(y_train, 'values'):  # pandas Series
        y_train = y_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'num_examples': len(X_train),
    }


def get_client(config: Dict, data: Tuple, client_id: int) -> fl.client.Client:
    """Create and return XGBoost federated learning client.
    
    Args:
        config: Configuration dictionary containing experiment settings
        data: Tuple of ((X_train, y_train), (X_test, y_test), time_col, event_col)
    
    Returns:
        Initialized XGBoostClient
    """
    
    (X_train, y_train), (X_test, y_test) = data
    
    # Convert to format expected by client
    local_data = get_numpy(X_train, y_train, X_test, y_test)
    
    # Create client
    client = XGBoostClient(
        local_data=local_data,
        client_id=client_id,
        saving_path=config.get("experiment_dir", "logs/sandbox/"),
        config=config
    )
    
    return client