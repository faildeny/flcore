
import time
import warnings

import flwr as fl
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import (train_test_split)

import flcore.models.linear_models.utils as utils
from flcore.metrics import calculate_metrics, find_best_threshold


# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, data,client_id,config):
        self.client_id = client_id
        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data

        # Create train and validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=config['seed'], stratify=self.y_train)

        self.model_name = config['model']
        self.config = config
        self.model = utils.get_model(self.model_name, self.config)
        self.round_time = 0
        self.first_round = True
        self.personalize = True
        self.use_fedprox = str(config.get("strategy", "FedAvg")).lower() == "fedprox"
        self.fedprox_mu = float(config.get("fedprox_mu", 0.02))
        self.fairness_attribute = config.get("parititon_by_attribute", None)
        if self.fairness_attribute is None:
            self.fairness_attribute = config.get("partition_by_attribute", None)
        self.fairness_attributes = (
            [self.fairness_attribute] if self.fairness_attribute is not None else None
        )
        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self.model, (self.X_train, self.y_train))

    def _predict_scores(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return self.model.decision_function(X)

    def _get_fairness_kwargs(self, X_subset):
        if self.fairness_attributes is None:
            return {}
        return {"X": X_subset, "fairness_attributes": self.fairness_attributes}
    
    def get_parameters(self, config):  # type: ignore
        #compute the feature selection
        #We perform it from the one called by the server
        #at the begining to start the parameters
        # if(bool(config) == False):
        #         fs = SelectKBest(f_classif, k= self.n_features).fit(self.X_train, self.y_train)
        #         index_features = fs.get_support()
        #         self.model.features = index_features
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        global_params = [np.copy(param) for param in parameters]
        utils.set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #To implement the center dropout, we need the execution time
            start_time = time.time()

            if self.use_fedprox and self.fedprox_mu > 0.0 and isinstance(self.model, SGDClassifier):
                local_steps = 10
                for _ in range(local_steps):
                    self.model.partial_fit(self.X_train, self.y_train)
                    # FedProx proximal correction on model parameters
                    self.model.coef_ = self.model.coef_ - self.fedprox_mu * (self.model.coef_ - global_params[0])
                    if self.model.fit_intercept:
                        self.model.intercept_ = self.model.intercept_ - self.fedprox_mu * (
                            self.model.intercept_ - global_params[1]
                        )
                print("Training with FedProx, mu =", self.fedprox_mu)
            else:
                print("Training without FedProx")
                self.model.fit(self.X_train, self.y_train)

            # self.model.fit(self.X_train.loc[:, parameters[2].astype(bool)], self.y_train)
            # y_pred = self.model.predict(self.X_test.loc[:, parameters[2].astype(bool)])
            y_pred_proba = self._predict_scores(self.X_val)
            best_threshold = find_best_threshold(self.y_val, y_pred_proba, metric="balanced_accuracy")
            y_pred_proba = self._predict_scores(self.X_test)
            metrics = calculate_metrics(
                self.y_test,
                y_pred_proba,
                threshold=best_threshold,
                **self._get_fairness_kwargs(self.X_test),
            )
            # Add 'personalized' to the metrics to identify them
            metrics = {f"personalized {key}": metrics[key] for key in metrics}
            self.round_time = (time.time() - start_time)
            metrics["running_time"] = self.round_time
            

        print(f"Training finished for round {config['server_round']}")

        if self.first_round:
            local_model = utils.get_model(self.model_name, self.config, local=True)
            # utils.set_initial_params(local_model,self.n_features)
            print("Training local model for comparison")
            local_model.fit(self.X_train, self.y_train)
            # Calculate validation set metrics
            y_pred_proba = self._predict_scores(self.X_val)
            best_threshold = find_best_threshold(self.y_val, y_pred_proba, metric="balanced_accuracy")
            y_pred_proba = self._predict_scores(self.X_test)
            local_metrics = calculate_metrics(
                self.y_test,
                y_pred_proba,
                threshold=best_threshold,
                **self._get_fairness_kwargs(self.X_test),
            )
            #Add 'local' to the metrics to identify them
            local_metrics = {f"local {key}": local_metrics[key] for key in local_metrics}
            metrics.update(local_metrics)
            metrics["client_id"] = self.client_id
            self.first_round = False

        return utils.get_model_parameters(self.model), len(self.X_train), metrics

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(self.model, parameters)

        # Calculate validation set metrics
        y_pred_proba = self._predict_scores(self.X_val)
        best_threshold = find_best_threshold(self.y_val, y_pred_proba, metric="balanced_accuracy")
        val_metrics = calculate_metrics(
            self.y_val,
            y_pred_proba,
            threshold=best_threshold,
            **self._get_fairness_kwargs(self.X_val),
        )

        y_pred_proba = self._predict_scores(self.X_test)
        # y_pred = self.model.predict(self.X_test.loc[:, parameters[2].astype(bool)])

        if(isinstance(self.model, SGDClassifier)):
            loss = 1.0
        else:
            loss = log_loss(self.y_test, self.model.predict_proba(self.X_test), labels=[0, 1])
       
        metrics = calculate_metrics(
            self.y_test,
            y_pred_proba,
            threshold=best_threshold,
            **self._get_fairness_kwargs(self.X_test),
        )
        metrics_not_tuned = calculate_metrics(
            self.y_test,
            y_pred_proba,
            threshold=0.5,
            **self._get_fairness_kwargs(self.X_test),
        )
        metrics_not_tuned = {f"not tuned {key}": metrics_not_tuned[key] for key in metrics_not_tuned}
        metrics.update(metrics_not_tuned)
        metrics["round_time [s]"] = self.round_time
        metrics["client_id"] = self.client_id

        # Add validation metrics to the evaluation metrics with a prefix
        val_metrics = {f"val {key}": val_metrics[key] for key in val_metrics}
        metrics.update(val_metrics)


        return loss, len(y_pred_proba),  metrics


def get_client(config,data,client_id) -> fl.client.Client:
    return MnistClient(data,client_id,config)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())

