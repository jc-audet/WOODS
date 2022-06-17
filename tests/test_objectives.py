
# import torch
# from torch import nn, optim
# from argparse import Namespace

# from woods import hyperparams, models, objectives
# from woods.train import train
# from woods.datasets import get_dataset_class, num_environments
# from woods.objectives import OBJECTIVES

# def test_objectives():

#     for objective in OBJECTIVES:

#         # Create dummy flags
#         flags = Namespace(
#             mode = 'train',
#             dataset = 'Spurious_Fourier',
#             test_env = 0,
#             holdout_fraction = 0.2,
#             objective = objective,
#             sample_hparams = False,
#             hparams_seed = 0,
#             trial_seed = 0,
#             data_path = '/hdd/dataset_test',
#             save_path = './results',
#             download = True,
#             save = False,
#             model_path = None)

#         # Get hparams
#         training_hparams = hyperparams.get_training_hparams(flags.dataset, flags.hparams_seed, flags.sample_hparams)
#         training_hparams['device'] = 'cpu'
#         objective_hparams = hyperparams.get_objective_hparams(flags.objective, flags.hparams_seed, flags.sample_hparams)
#         model_hparams = hyperparams.get_model_hparams(flags.dataset)

#         # Create dataset
#         dataset_class = get_dataset_class('Spurious_Fourier')
#         dataset = dataset_class(flags, training_hparams)

#         # Set the dataset n_steps to 2
#         dataset.N_STEPS = 2

#         # Initialize model
#         model = models.get_model(dataset, model_hparams)
#         print("Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

#         # Define training aid
#         # loss_fn = nn.NLLLoss(weight=dataset.get_class_weight().to(device))
#         loss_fn = dataset.loss_fn
#         optimizer = optim.Adam(model.parameters(), lr=training_hparams['lr'], weight_decay=training_hparams['weight_decay'])

#         ## Initialize some Objective
#         objective_class = objectives.get_objective_class(flags.objective)
#         objective = objective_class(model, dataset, loss_fn, optimizer, objective_hparams)

#         # Run test
#         model, record, table = train(flags, training_hparams, model, objective, dataset, 'cpu')
