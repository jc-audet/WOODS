import numpy as np

from woods.lib.objectives import OBJECTIVES

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


def Fourier_basic_train(sample):
    """ Spurious Fourier model hparam definition """
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
            'batch_size': lambda r: 64
        }

    return hparams
    
def Spurious_Fourier_train(sample):
    """ Spurious Fourier model hparam definition """
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
            'batch_size': lambda r: 64
        }

    return hparams

def TMNIST_train(sample):
    """ TMNIST model hparam definition """
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
            'batch_size': lambda r: 64
        }

    return hparams

def TCMNIST_seq_train(sample):
    """ TCMNIST_seq model hparam definition """
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
            'batch_size': lambda r: 64
        }

    return hparams

def TCMNIST_step_train(sample):
    """ TCMNIST_step model hparam definition """
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
            'batch_size': lambda r: 64
        }

    return hparams

def CAP_DB_train(sample):
    """ PhysioNet model hparam definition """
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
            'batch_size': lambda r: 8
        }

    return hparams

def SEDFx_DB_train(sample):
    """ PhysioNet model hparam definition """
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
            'batch_size': lambda r: 8
        }

    return hparams


def MI_train(sample):
    """ MI model hparam definition """
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
            'weight_decay': lambda r: 0.1,
            'lr': lambda r: 10**-4,
            'batch_size': lambda r: 8
        }

    return hparams


def HAR_train(sample):
    """ PhysioNet model hparam definition """
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
            'batch_size': lambda r: 8
        }

    return hparams

def LSA64_train(sample):
    """ PhysioNet model hparam definition """
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
            'batch_size': lambda r: 8
        }

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

def Fourier_basic_model(sample):
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

def TMNIST_model(sample):
    """ TMNIST model hparam definition """
    if sample:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: int(r.choice([1, 2, 3])),
            'hidden_width': lambda r: int(2**r.uniform(5, 7)),
            'recurrent_layers': lambda r: int(r.choice([1, 2, 3])),
            'state_size': lambda r: int(2**r.uniform(5, 7))
        }
    else:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: 1, 
            'hidden_width': lambda r: 20,
            'recurrent_layers': lambda r: 2,
            'state_size': lambda r: 32
        }

def TCMNIST_seq_model(sample):
    """ TCMNIST_seq model hparam definition """
    if sample:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: int(r.choice([1, 2, 3])),
            'hidden_width': lambda r: int(2**r.uniform(5, 7)),
            'recurrent_layers': lambda r: int(r.choice([1, 2, 3])),
            'state_size': lambda r: int(2**r.uniform(5, 7))
        }
    else:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: 1, 
            'hidden_width': lambda r: 20,
            'recurrent_layers': lambda r: 2,
            'state_size': lambda r: 32
        }

def TCMNIST_step_model(sample):
    """ TCMNIST_step model hparam definition """
    if sample:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: int(r.choice([1, 2, 3])),
            'hidden_width': lambda r: int(2**r.uniform(5, 7)),
            'recurrent_layers': lambda r: int(r.choice([1, 2, 3])),
            'state_size': lambda r: int(2**r.uniform(5, 7))
        }
    else:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: 1, 
            'hidden_width': lambda r: 20,
            'recurrent_layers': lambda r: 2,
            'state_size': lambda r: 32
        }

def CAP_DB_model(sample):
    """ PhysioNet model hparam definition """
    if sample:
        return {
            'model': lambda r: 'Transformer',
            'nheads_enc': lambda r: 8,
            'nlayers_enc': lambda r: 2,
            'embedding_size': lambda r: 32
        }
    else:
        return {
            'model': lambda r: 'Transformer',
            'nheads_enc': lambda r: 8,
            'nlayers_enc': lambda r: 2,
            'embedding_size': lambda r: 32
        }

def SEDFx_DB_model(sample):
    """ PhysioNet model hparam definition """
    if sample:
        return {
            'model': lambda r: 'Transformer',
            'nheads_enc': lambda r: 8,
            'nlayers_enc': lambda r: 2,
            'embedding_size': lambda r: 32
        }
    else:
        return {
            'model': lambda r: 'Transformer',
            'nheads_enc': lambda r: 8,
            'nlayers_enc': lambda r: 2,
            'embedding_size': lambda r: 32
        }


def MI_model(sample):
    """ MI model hparam definition """
    if sample:
        return {
            'model': lambda r: 'Transformer',
            'nheads_enc': lambda r: 8,
            'nlayers_enc': lambda r: 2,
            'embedding_size': lambda r: 32
        }
    else:
        return {
            'model': lambda r: 'shallow',
            'nheads_enc': lambda r: 8,
            'nlayers_enc': lambda r: 2,
            'embedding_size': lambda r: 32
        }


# def HAR_model(sample):
#     """ PhysioNet model hparam definition """
#     if sample:
#         return {
#             'model': lambda r: 'Transformer',
#             'nheads_enc': lambda r: 8,
#             'nlayers_enc': lambda r: 2,
#             'embedding_size': lambda r: 32
#         }
#     else:
#         return {
#             'model': lambda r: 'Transformer',
#             'nheads_enc': lambda r: 8,
#             'nlayers_enc': lambda r: 2,
#             'embedding_size': lambda r: 32
#         }
def HAR_model(sample):
    """ TCMNIST_seq model hparam definition """
    if sample:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: int(r.choice([1, 2, 3])),
            'hidden_width': lambda r: int(2**r.uniform(5, 7)),
            'recurrent_layers': lambda r: int(r.choice([1, 2, 3])),
            'state_size': lambda r: int(2**r.uniform(5, 7))
        }
    else:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: 1, 
            'hidden_width': lambda r: 20,
            'recurrent_layers': lambda r: 2,
            'state_size': lambda r: 32
        }

def LSA64_model(sample):
    """ TCMNIST_seq model hparam definition """
    if sample:
        return {
            'model': lambda r: 'LSTM',
            'hidden_depth': lambda r: int(r.choice([1, 2, 3])),
            'hidden_width': lambda r: int(2**r.uniform(5, 7)),
            'recurrent_layers': lambda r: int(r.choice([1, 2, 3])),
            'state_size': lambda r: int(2**r.uniform(5, 7))
        }
    else:
        return {
            'model': lambda r: 'CRNN',
            # Classifier
            'hidden_depth': lambda r: 1,
            'hidden_width': lambda r: 64,
            # LSTM
            'recurrent_layers': lambda r: 2,
            'state_size': lambda r: 128,
            # Resnet encoder
            'fc_hidden': lambda r: (512,512),
            'CNN_embed_dim': lambda r: 256
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
