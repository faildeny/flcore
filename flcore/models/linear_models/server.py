#############################################################################
#Linear models implemented by Esmeralda Ruiz Pujadas                       ##
#The Linear Models are: LR, ElasticNet and LSVM                            ##
#You can select them from the params of config                             ##
#It is implemented with feature selection                                  ##
#In this implementation the first client selected by the server provides   ##
#the feature selection and is sent to the server and the server sends it   ##
#to all the clients as all the clients must use the same features          ##
#Feel free to implement more sophisticated feature selection               ##
#To disable the feature selection select the maximum features and all the  ##
#features will be used using n_features in config                          ##
#Params in config:                                                         ##
# Type: elastic_net,LSVC, LR                                               ##
# num_features                                                             ##
#Mising: Pipeline to deal with categorical                                 ##
#############################################################################

import warnings
from typing import Dict

from flcore.metrics import metrics_aggregation_fn
from flcore.models.linear_models.FedCustomAggregator import FedCustom


warnings.filterwarnings( 'ignore' )

def fit_round( server_round: int ) -> Dict:
    """Send round number to client."""
    return { 'server_round': server_round }


def get_server_and_strategy(config):
    # Pass parameters to the Strategy for server-side parameter initialization
    strategy = FedCustom(   
        min_available_clients = config['num_clients'],
        min_fit_clients = config['num_clients'],
        min_evaluate_clients = config['num_clients'],
        fit_metrics_aggregation_fn = metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn = metrics_aggregation_fn,
        on_fit_config_fn = fit_round,
        checkpoint_dir = config["experiment_dir"] / "checkpoints",
        dropout_method = config['dropout_method'],
        percentage_drop = config['dropout']['percentage_drop'],
        smoothing_method = config['smooth_method'],
        smoothing_strenght = config['smoothWeights']['smoothing_strenght']
    )

    return None, strategy
