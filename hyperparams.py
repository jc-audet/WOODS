import numpy as np

from objectives import OBJECTIVES

def get_training_hparams(seed, sample=False):

    if sample:
        hparams = {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-4.5, -2.5),
            'batch_size': lambda r: int(2**r.uniform(3, 9))
        }
    else:
        hparams = {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 1e-3,
            'batch_size': lambda r: 6
        }
    
    for k in hparams.keys():
        hparams[k] = hparams[k](np.random.RandomState(seed))

    return hparams

## TODO: maybe the dataset hparams should be a part of the dataset object?
def get_model_hparams(dataset_name, seed, sample=False):

    """Return the dataset class with the given name."""
    dataset_model = dataset_name + '_model'
    if dataset_model not in globals():
        raise NotImplementedError("dataset not found: {}".format(dataset_name))
    else:
        hyper_function = globals()[dataset_model]

    hparams = hyper_function(sample)

    for k in hparams.keys():
        hparams[k] = hparams[k](np.random.RandomState(seed))
    
    return hparams

def Spurious_Fourier_model(sample):

    if sample:
        return {
            'model': lambda r: 'RNN',
            'hidden_depth': lambda r: int(r.choice([0, 1, 2])),
            'hidden_width': lambda r: int(2**r.uniform(5, 7)),
            'state_size': lambda r: 10
        }
    else:
        return {
            'model': lambda r: 'RNN',
            'hidden_depth': lambda r: 0,
            'hidden_width': lambda r: 20,
            'state_size': lambda r: 10
        }

def TCMNIST_seq_model(sample):

    if sample:
        return {
            'model': lambda r: 'RNN',
            'hidden_depth': lambda r: int(r.choice([2, 3, 4])),
            'hidden_width': lambda r: int(2**r.uniform(5, 9)),
            'state_size': lambda r: 10
        }
    else:
        return {
            'model': lambda r: 'RNN',
            'hidden_depth': lambda r: 2, 
            'hidden_width': lambda r: 20,
            'state_size': lambda r: 10
        }

def TCMNIST_step_model(sample):

    if sample:
        return {
            'model': lambda r: 'RNN',
            'hidden_depth': lambda r: r.choice([2, 3, 4]),
            'hidden_width': lambda r: int(2**r.uniform(5, 9)),
            'state_size': lambda r: 10
        }
    else:
        return {
            'model': lambda r: 'RNN',
            'hidden_depth': lambda r: 2, 
            'hidden_width': lambda r: 20,
            'state_size': lambda r: 10
        }

def PhysioNet_model(sample):

    if sample:
        return {
            'model': lambda r: 'ATTN_LSTM',
            'hidden_depth': lambda r: int(r.choice([2, 3, 5])),
            'hidden_width': lambda r: int(2**r.uniform(7, 10)),
            'recurrent_layers': lambda r: int(r.choice([2, 3, 5])),
            'state_size': lambda r: int(2**r.uniform(7, 11))
        }
    else:
        return {
            'model': lambda r: 'ATTN_LSTM',
            'hidden_depth': lambda r: 2,
            'hidden_width': lambda r: 256,
            'recurrent_layers': lambda r: 2,
            'state_size': lambda r: 256
        }


def get_objective_hparams(objective_name, seed, sample=False):

    """Return the objective class with the given name."""
    objective_hyper = objective_name+'_hyper'
    if objective_hyper not in globals():
        raise NotImplementedError("objective not found: {}".format(objective_name))
    else:
        hyper_function = globals()[objective_hyper]

    hparams = hyper_function(sample)

    for k in hparams.keys():
        hparams[k] = hparams[k](np.random.RandomState(seed))
    
    return hparams

def ERM_hyper(sample):

    return {}

def IRM_hyper(sample):

    if sample:
        return {
            'penalty_weight': lambda r: 10**r.uniform(-1,5),
            'anneal_iters': lambda r: r.uniform(0,2000)
        }
    else:
        return {
            'penalty_weight': lambda r: 1e4,
            'anneal_iters': lambda r: 10
        }

def VREx_hyper(sample):

    if sample:
        return {
            'penalty_weight': lambda r: 10**r.uniform(-1,5),
            'anneal_iters': lambda r: r.uniform(0,2000)
        }
    else:
        return {
            'penalty_weight': lambda r: 1e4,
            'anneal_iters': lambda r: 10
        }

def SD_hyper(sample):

    if sample:
        return {
            'penalty_weight': lambda r: 10**r.uniform(-2,2)
        }
    else:
        return {
            'penalty_weight': lambda r: 1
        }
        
def IGA_hyper(sample):

    if sample:
        return {
            'penalty_weight': lambda r: 10**r.uniform(-1,5)
        }
    else:
        return {
            'penalty_weight': lambda r: 1e4
        }

def ANDMask_hyper(sample):

    if sample:
        return {
            'tau': lambda r: r.uniform(0,1)
        }
    else:
        return {
            'tau': lambda r: 1
        }