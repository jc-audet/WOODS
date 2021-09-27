import numpy as np

from lib.objectives import OBJECTIVES

# TODO : Check config files

def get_training_hparams(seed, sample=False):
    """ Get training related hyper parameters (class_balance, weight_decay, lr, batch_size)

    Args:
        seed (int): seed used if hyper parameter is sampled
        sample (bool, optional): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.

    Returns:
        dict: Dictionnary with hyper parameters values
    
    TODO:
        * This should be defined as a function of the dataset used.
    """
    if sample:
        hparams = {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-4.5, -2.5),
            'batch_size': lambda r: int(2**r.uniform(3, 7))
        }
    else:
        hparams = {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 10**-4.5,
            'batch_size': lambda r: 32
        }
    
    for k in hparams.keys():
        hparams[k] = hparams[k](np.random.RandomState(seed))

    return hparams

def get_model_hparams(dataset_name, seed, sample=False):
    """ Get the model related hyper parameters

    Each dataset has their own model hyper parameters definition

    Args:
        dataset_name (str): dataset that is gonna be trained on for the run
        seed (int): seed used if hyper parameter is sampled
        sample (bool, optional): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.

    Raises:
        NotImplementedError: Dataset name not found

    Returns:
        dict: Dictionnary with hyper parameters values
    """
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
    """ Spurious Fourier model hparam definition """
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
    """ TCMNIST_seq model hparam definition """
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
    """ TCMNIST_step model hparam definition """
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

# def PhysioNet_model(sample):
#     """ PhysioNet model hparam definition """
#     if sample:
#         return {
#             'model': lambda r: 'ATTN_LSTM',
#             'hidden_depth': lambda r: int(r.choice([2, 3, 5])),
#             'hidden_width': lambda r: int(2**r.uniform(7, 10)),
#             'recurrent_layers': lambda r: int(r.choice([2, 3, 5])),
#             'state_size': lambda r: int(2**r.uniform(7, 11))
#         }
#     else:
#         return {
#             'model': lambda r: 'ATTN_LSTM',
#             'hidden_depth': lambda r: 2,
#             'hidden_width': lambda r: 256,
#             'recurrent_layers': lambda r: 2,
#             'state_size': lambda r: 256
#         }

def PhysioNet_model(sample):
    """ PhysioNet model hparam definition """
    if sample:
        return {
            'model': lambda r: 'Transformer'
        }
    else:
        return {
            'model': lambda r: 'Transformer',
            'nheads_enc': lambda r: 8,
            'nlayers_enc': lambda r: 2,
            'embedding_size': lambda r: 32
        }


def get_objective_hparams(objective_name, seed, sample=False):
    """ Get the objective related hyper parameters

    Each objective has their own model hyper parameters definitions

    Args:
        objective_name (str): objective that is gonna be trained on for the run
        seed (int): seed used if hyper parameter is sampled
        sample (bool, optional): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.

    Raises:
        NotImplementedError: Objective name not found

    Returns:
        dict: Dictionnary with hyper parameters values
    """
    # Return the objective class with the given name
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
    """ ERM objective hparam definition """
    return {}

def IRM_hyper(sample):
    """ IRM objective hparam definition """
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
    """ VREx objective hparam definition """
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
    """ SD objective hparam definition """
    if sample:
        return {
            'penalty_weight': lambda r: 10**r.uniform(-2,2)
        }
    else:
        return {
            'penalty_weight': lambda r: 1
        }
        
def IGA_hyper(sample):
    """ IGA objective hparam definition """
    if sample:
        return {
            'penalty_weight': lambda r: 10**r.uniform(-1,5)
        }
    else:
        return {
            'penalty_weight': lambda r: 1e4
        }

def ANDMask_hyper(sample):
    """ ANDMask objective hparam definition """
    if sample:
        return {
            'tau': lambda r: r.uniform(0,1)
        }
    else:
        return {
            'tau': lambda r: 1
        }
