"""Defining hyper parameters and their distributions for HPO"""

import numpy as np

from woods.objectives import OBJECTIVES

def get_training_hparams(dataset_name, seed, sample=False):
    """ Get training related hyper parameters (class_balance, weight_decay, lr, batch_size)

    Args:
        dataset_name (str): dataset that is gonna be trained on for the run
        seed (int): seed used if hyper parameter is sampled
        sample (bool, optional): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.

    Raises:
        NotImplementedError: Dataset name not found

    Returns:
        dict: Dictionnary with hyper parameters values
    """

    dataset_train = dataset_name + '_train'
    if dataset_train not in globals():
        raise NotImplementedError("dataset not found: {}".format(dataset_name))
    else:
        hyper_function = globals()[dataset_train]

    hparams = hyper_function(sample)
    
    for k in hparams.keys():
        hparams[k] = hparams[k](np.random.RandomState(seed))

    return hparams


def Basic_Fourier_train(sample):
    """ Basic Fourier model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-4.5, -2.5),
            'batch_size': lambda r: int(2**r.uniform(3, 7))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 1e-3,
            'batch_size': lambda r: 64
        }
    
def Spurious_Fourier_train(sample):
    """ Spurious Fourier model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-4.5, -2.5),
            'batch_size': lambda r: int(2**r.uniform(3, 7))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 1e-3,
            'batch_size': lambda r: 64
        }

def TMNIST_train(sample):
    """ TMNIST model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-4.5, -2.5),
            'batch_size': lambda r: int(2**r.uniform(3, 9))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 1e-3,
            'batch_size': lambda r: 64
        }

def TCMNIST_seq_train(sample):
    """ TCMNIST_seq model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-4.5, -2.5),
            'batch_size': lambda r: int(2**r.uniform(3, 9))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 1e-3,
            'batch_size': lambda r: 64
        }

def TCMNIST_step_train(sample):
    """ TCMNIST_step model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-4.5, -2.5),
            'batch_size': lambda r: int(2**r.uniform(3, 9))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 1e-3,
            'batch_size': lambda r: 64
        }

def CAP_train(sample):
    """ CAP model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-5, -3),
            'batch_size': lambda r: int(2**r.uniform(3, 4))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 10**-4,
            'batch_size': lambda r: 8
        }

def SEDFx_train(sample):
    """ SEDFx model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-5, -3),
            'batch_size': lambda r: int(2**r.uniform(3, 4))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 10**-4,
            'batch_size': lambda r: 8
        }

def MI_train(sample):
    """ MI model hparam definition """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-5, -3),
            'batch_size': lambda r: int(2**r.uniform(3, 4))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**-3,
            'batch_size': lambda r: 8
        }

def HAR_train(sample):
    """ HAR model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-5, -3),
            'batch_size': lambda r: int(2**r.uniform(3, 4))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 10**-4,
            'batch_size': lambda r: 8
        }

def LSA64_train(sample):
    """ LSA64 model hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-5, -3),
            'batch_size': lambda r: int(2**r.uniform(3, 4))
        }
    else:
        return {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 10**-4,
            'batch_size': lambda r: 8
        }

def StockVolatility_train(sample):
    """ StockVolatility hparam definition """
    if sample:
        hparams = {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0.,
            'lr': lambda r: 10**r.uniform(-5, -3),
            'batch_size': lambda r: int(2**r.uniform(3, 4))
        }
    else:
        hparams = {
            'class_balance': lambda r: True,
            'weight_decay': lambda r: 0,
            'lr': lambda r: 10**-4,
            'batch_size': lambda r: 32
        }

    return hparams

def get_model_hparams(dataset_name):
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

    hparams = hyper_function()
    
    return hparams

def Basic_Fourier_model():
    """ Spurious Fourier model hparam definition """
    return {
        'model': 'LSTM',
        'hidden_depth': 1, 
        'hidden_width': 20,
        'recurrent_layers': 2,
        'state_size': 32
    }

def Spurious_Fourier_model():
    """ Spurious Fourier model hparam definition """
    return {
        'model': 'LSTM',
        'hidden_depth': 1, 
        'hidden_width': 20,
        'recurrent_layers': 2,
        'state_size': 32
    }

def TMNIST_model():
    """ TMNIST model hparam definition """
    return {
        'model': 'MNIST_LSTM',
        'hidden_depth': 3, 
        'hidden_width': 64,
        'recurrent_layers': 1,
        'state_size': 128
    }

def TCMNIST_seq_model():
    """ TCMNIST_seq model hparam definition """
    return {
        'model': 'MNIST_LSTM',
        'hidden_depth': 3, 
        'hidden_width': 64,
        'recurrent_layers': 1,
        'state_size': 128
    }

def TCMNIST_step_model():
    """ TCMNIST_step model hparam definition """
    return {
        'model': 'MNIST_LSTM',
        'hidden_depth': 3, 
        'hidden_width': 64,
        'recurrent_layers': 1,
        'state_size': 128
    }

def CAP_model():
    """ CAP model hparam definition """
    return {
        'model': 'deep4'
    }

def SEDFx_model():
    """ SEDFx model hparam definition """
    return {
        'model': 'deep4'
    }

def MI_model():
    """ MI model hparam definition"""
    return {
        'model': 'deep4'
    }

def HAR_model():
    """ HAR model hparam definition """
    return {
        'model': 'deep4',
    }

def LSA64_model():
    """ LSA64 model hparam definition """
    return {
        'model': 'CRNN',
        # Classifier
        'hidden_depth': 1,
        'hidden_width': 64,
        # LSTM
        'recurrent_layers': 2,
        'state_size': 128,
        # Resnet encoder
        'fc_hidden': (512,512),
        'CNN_embed_dim': 256
    }

def StockVolatility_model():
    """ StockVolatility model hparam definition """
    return {
        'model': 'LSTM',
        'hidden_depth': 1, 
        'hidden_width': 20,
        'recurrent_layers': 2,
        'state_size': 32
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
    """ ERM objective hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    return {}

def IRM_hyper(sample):
    """ IRM objective hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
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
    """ VREx objective hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
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
    """ SD objective hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'penalty_weight': lambda r: 10**r.uniform(-2,2)
        }
    else:
        return {
            'penalty_weight': lambda r: 1
        }
        
def IGA_hyper(sample):
    """ IGA objective hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'penalty_weight': lambda r: 10**r.uniform(-1,5)
        }
    else:
        return {
            'penalty_weight': lambda r: 1e4
        }

def ANDMask_hyper(sample):
    """ ANDMask objective hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'tau': lambda r: r.uniform(0,1)
        }
    else:
        return {
            'tau': lambda r: 1
        }


def Fish_hyper(sample):
    """ Fish objective hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'meta_lr': lambda r: 0.5
        }
    else:
        return {
            'meta_lr': lambda r:r.choice([0.05, 0.1, 0.5])
        }
        
        
        
def SANDMask_hyper(sample):
    """ SANDMask objective hparam definition 
    
    Args:
        sample (bool): If ''True'', hyper parameters are gonna be sampled randomly according to their given distributions. Defaults to ''False'' where the default value is chosen.
    """
    if sample:
        return {
            'tau': lambda r: r.uniform(0.0,1.),
            'k': lambda r: 10**r.uniform(-3, 5),
            'betas': lambda r: r.uniform(0.9,0.999)
        }
    else:
        return {
            'tau': lambda r: 1,
            'k': lambda r: 1e+1,
            'betas': lambda r: 0.9
        }        

