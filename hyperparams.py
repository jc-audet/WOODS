import numpy as np

from objectives import OBJECTIVES


def get_hyperparams(objective_name, seed, sample=False):

    """Return the objective class with the given name."""
    objective_hyper = objective_name+'_hyper'
    if objective_hyper not in globals():
        raise NotImplementedError("objective not found: {}".format(objective_name))
    else:
        hyper_function = globals()[objective_hyper]

    hyper_dict = hyper_function(sample)

    for k in hyper_dict.keys():
        hyper_dict[k] = hyper_dict[k](np.random.RandomState(seed))
    
    return hyper_dict

def ERM_hyper(sample):

    return {}


def IRM_hyper(sample):

    if sample:
        return {
            'penalty_weight': lambda r: 10**r.uniform(-1,5),
            'anneal_iters': lambda r: r.uniform(0,100)
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
            'anneal_iters': lambda r: r.uniform(0,100)
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

def SANDMask_hyper(sample):

    if sample:
        return {
            'tau': lambda r: r.uniform(0,1),
            'k': lambda r: 10**r.uniform(-4,4)
        }
    else:
        return {
            'tau': lambda r: 1,
            'k': lambda r: 10
        }