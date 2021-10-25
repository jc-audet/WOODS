import os
import json
import tqdm
import numpy as np
from argparse import Namespace
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from scripts import hparams_sweep
from lib import datasets

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_results(results_path):
    """ Plot results - accuracy and loss - w.r.t. training step

    Args:
        results_path (str): path to a results json file coming from a training run
    """
    
    # Fetch data
    with open(results_path, 'r') as fp:
        results = json.load(fp)
    
    # Aggregate results
    results_arrs = {}
    steps = [ key for key in results.keys() if key not in ['hparams', 'flags']]
    for s in steps:
        for split in results[s].keys():
            try:
                results_arrs[split].append(results[s][split])
            except KeyError:
                results_arrs[split] = []
                results_arrs[split].append(results[s][split])
    
    # Get keys
    loss_keys = []
    acc_keys = []
    steps = [int(s) for s in steps]
    for k in results_arrs.keys():
        if 'loss' in k:
            loss_keys.append(k)
        if 'acc' in k:
            acc_keys.append(k)

    # Get environment names
    envs = datasets.get_environments(results['flags']['dataset'])
    test_env = envs[results['flags']['test_env']]
    env_color = get_cmap(len(envs), name='jet')

    # Plot loss
    plt.figure()
    for i, e in enumerate(envs):
        if e == test_env:
            linewidth = 2
            label = e + '(test)'
        else:
            linewidth = 1
            label = e

        plt.plot(steps, results_arrs[e+'_in_loss'], color = env_color(i), linestyle='-', label=label, linewidth=linewidth)
        plt.plot(steps, results_arrs[e+'_out_loss'], color = env_color(i), linestyle='--', linewidth=linewidth)
    plt.legend()

    # Plot accuracy
    plt.figure()
    for i, e in enumerate(envs):
        if e == test_env:
            linewidth = 2
        else:
            linewidth = 1

        plt.plot(steps, results_arrs[e+'_in_acc'], color = env_color(i), linestyle='-', label=label, linewidth=linewidth)
        plt.plot(steps, results_arrs[e+'_out_acc'], color = env_color(i), linestyle='--', linewidth=linewidth)
    plt.legend()
    plt.show()

def print_results(results_path):
    """ Print results from a results json file
    Args:
        results_path (str): path to a results json file coming from a training run
    """

    # Fetch the data
    with open(results_path, 'r') as fp:
        results = json.load(fp)

    print('Flags:')
    for k, v in sorted(results['flags'].items()):
        print('\t{}: {}'.format(k, v))
    print('HParams:')
    for k, v in sorted(results['hparams'].items()):
        print('\t{}: {}'.format(k, v))


    # Setup the PrettyTable from printing
    t = setup_pretty_table(Namespace(**results['flags']))

    # Get env names
    envs = datasets.get_environments(results['flags']['dataset'])
    test_env = envs[results['flags']['test_env']]
    
    train_names = [str(env)+"_in_loss" for i, env in enumerate(envs) if i != results['flags']['test_env']]
    
    # Go through checkpoint step by checkpoint step and append to the table
    steps = [ key for key in results.keys() if key not in ['hparams', 'flags']]
    for s in steps:
        t.add_row([s] 
                + ["{:.2f} :: {:.2f}".format(results[s][str(e)+'_in_acc'], results[s][str(e)+'_out_acc']) for e in envs] 
                + ["{:.2f}".format(np.average([results[s][str(e)] for e in train_names]))] 
                + ["{}".format('.')]
                + ["{}".format('.')]
                + ["{}".format('.')] )

        print("\n".join(t.get_string().splitlines()[-2:-1]))
    
def get_job_name(flags):
    """ Generates the name of the output file for a training run as a function of the config

    Seq setup:
    <objective>_<dataset>_<test_env>_H<hparams_seed>_T<trial_seed>.json
    Step setup:
    <objective>_<dataset>_<test_env>_H<hparams_seed>_T<trial_seed>_S<test_step>.json

    Args:
        flags (dict): dictionnary of the config for a training run

    Returns:
        str: name of the output json file of the training run 
    """

    job_id = flags['objective'] + '_' + flags['dataset'] + '_' + str(flags['test_env']) + '_H' + str(flags['hparams_seed']) + '_T' + str(flags['trial_seed'])

    return job_id

def check_file_integrity(results_dir):
    """ Check for integrity of files from a hyper parameter sweep

    Args:
        results_dir (str): directory where sweep results are stored

    Raises:
        AssertionError: If there is a sweep file missing
    """

    with open(os.path.join(results_dir,'sweep_config.json'), 'r') as fp:
        flags = json.load(fp)
    
    # Recall sweep config
    _, train_args = hparams_sweep.make_args_list(flags)

    # Check for sweep output files
    missing_files = 0
    for args in tqdm.tqdm(train_args, desc="Checking file integrity"):
        name = get_job_name(args) + '.json'
        
        if not os.path.exists(os.path.join(results_dir, name)):
            missing_files += 1

    assert missing_files == 0, str(missing_files) + " sweep results are missing from the results directory"

def setup_pretty_table(flags):
    """ Setup the printed table that show the results at each checkpoints

    Args:
        flags (Namespace): Namespace of the argparser containing the config of the training run
        dataset (Multi_Domain_Dataset): Dataset Object

    Returns:
        PrettyTable: an instance of prettytable.PrettyTable
    """

    job_id = 'Training ' + flags.objective  + ' on ' + flags.dataset + ' (H=' + str(flags.hparams_seed) + ', T=' + str(flags.trial_seed) + ')'

    t = PrettyTable()

    env_name = datasets.get_environments(flags.dataset)
    setup = datasets.get_setup(flags.dataset)

    if setup == 'seq':
        t.field_names = ['Env'] + [str(e) if i != flags.test_env else '** ' + str(e) + ' **' for i, e in enumerate(env_name)] + [' ', '  ', '   ', '    ']
    if setup == 'step':
        t.field_names = ['Env'] + [str(e) if i != len(env_name)-1 else '** ' + str(e) + ' **' for i, e in enumerate(env_name)] + [' ', '  ', '   ', '    ']

    max_width = {}
    min_width = {}
    for n in t.field_names:
        max_width.update({n: 15})
        min_width.update({n: 15})
    t._min_width = min_width
    t._max_width = max_width
    t.add_row(['Steps'] + ['in   :: out' for e in env_name] + ['Avg Loss', 'Epoch', 'Step Time', 'Val Time'])
    print(t.get_string(title=job_id, border=True, hrule=0))
    t.del_row(0)
    
    return t

def get_latex_table(table, caption=None, label=None):
    """Construct and export a LaTeX table from a PrettyTable.

    Inspired from : https://github.com/adasilva/prettytable

    Args:
        table (PrettyTable); an instance of prettytable.PrettyTable
        caption (str, optional): a caption for the table. Defaults to None.
        label (str, optional): a latex reference tag. Defaults to None.

    Returns:
        str: printable latex string
    """

    s = r'\begin{table}' + '\n'
    s = s + r'\centering' + '\n'
    s = s + r'\caption{%s}\label{%s}' %(caption, label)
    s = s + '\n'
    s = s + r'\begin{tabular}{'
    s = s + ''.join(['c',]*len(table.field_names)) + '}'
    s = s + '\n'
    s = s + ' & '.join(table.field_names)+r' \\ \hline'+'\n'
    rows = table._format_rows(table._rows, [])
    for i in range(len(rows)):
        row = [str(itm) for itm in rows[i]]
        s = s + ' & '.join(row)
        if i != len(table._rows)-1:
            s = s + r'\\'
        s = s + '\n'
        
    s = s + r'\end{tabular}' + '\n'
    s = s + r'\end{table}'
    return s
