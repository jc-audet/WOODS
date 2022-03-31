
import torch
from argparse import Namespace

import woods
from woods.hyperparams import get_training_hparams
from woods.datasets import get_dataset_class, num_environments
from woods.datasets import DATASETS, InfiniteLoader

def test_dataset_attributes():

    for dataset in DATASETS:

        dataset_class = get_dataset_class(dataset)

        assert dataset_class.ENVS is not None
        assert dataset_class.N_STEPS is not None
        assert dataset_class.CHECKPOINT_FREQ is not None
        assert dataset_class.N_WORKERS is not None
        assert dataset_class.SETUP is not None
        assert dataset_class.TASK is not None
        assert dataset_class.SEQ_LEN is not None
        assert dataset_class.PRED_TIME is not None
        assert dataset_class.INPUT_SHAPE is not None
        assert dataset_class.OUTPUT_SIZE is not None
        assert dataset_class.SWEEP_ENVS is not None

# def test_dataset_loaders():

#     for dataset_name in DATASETS:

#         if num_environments(dataset_name) > 1:
#             test_env = 0
#         else:
#             test_env = None

#         # Create dummy flags
#         flags = Namespace(
#             mode = 'train',
#             dataset = dataset_name,
#             test_env = test_env,
#             holdout_fraction = 0.2,
#             objective = 'ERM',
#             sample_hparams = False,
#             hparams_seed = 0,
#             trial_seed = 0,
#             data_path = '/hdd/dataset_test',
#             save_path = './results',
#             download = True,
#             save = False,
#             model_path = None)

#         # Get hparams
#         training_hparams = get_training_hparams(flags.dataset, flags.hparams_seed, flags.sample_hparams)
#         training_hparams['device'] = 'cpu'

#         # Create dataset
#         dataset_class = get_dataset_class(dataset_name)
#         dataset = dataset_class(flags, training_hparams)

#         # Make checks
#         for fast_loader in dataset.val_loaders:
#             assert isinstance(fast_loader, torch.utils.data.DataLoader)
#         for infinite_loader in dataset.train_loaders:
#             assert isinstance(infinite_loader, InfiniteLoader)
        


        