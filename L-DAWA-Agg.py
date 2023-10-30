import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

def my_aggregate(results: List[Tuple[Weights, float]], w_g):
    d = np.arange(len(w_g)) # numpy array equal to the length of layers
    c = [weights for weights, _ in results] # pass: it will give the weights of the clients as a list
    total_clients = len(results)
    delta_c = {} # dictionary to save the clients cosine between the global model and the
    for wi in range(len(c)):
        delta_ = []
        for x, y, id in zip(c[wi], w_g, d): # for each layer
            if len(x.shape) and len(y.shape) > 0: # remove the empty values
                if np.linalg.norm(x) > 0: # if it is zero
                    v = (x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y)) # computing the angular divergence 
                    # the "1." is multiplied below because python is unable to seralized values of 32bit float; multiplicaiton --> auto convert to 64bit float
                    delta_.append((str(id), 1.*v)) # store the value as a tuple in temporary list
                    if v > 1: 
                        v = 1.0 # the value cannot be greater than 1; sometimes the value overflow but not often
                        c[wi][id] = x*v # no change in the layer value 
                    else:
                        # multiply the layer by the cosine value
                        c[wi][id] =  x*v 
            else:
                c[wi][id] = c[wi][id] # retain the actual value; e.g tracking mean, tracking var if available
        delta_c[str(wi)] = delta_ 
    # after computing the divergence apply the averaging
    weights_prime: Weights = [
        reduce(np.add, layer_updates) / total_clients
        for layer_updates in zip(*c)
    ]   
    return weights_prime, delta_c
