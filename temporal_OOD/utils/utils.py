import os
import json
import tqdm
from argparse import Namespace
from prettytable import PrettyTable

from temporal_OOD.scripts import hparams_sweep

def get_job_json(flags):

    if flags.test_step is not None:
        job_id = flags.objective + '_' + flags.dataset + '_' + str(flags.test_env) + '_H' + str(flags.hparams_seed) + '_T' + str(flags.trial_seed) + '_S' + str(flags.test_step)
    else:
        job_id = flags.objective + '_' + flags.dataset + '_' + str(flags.test_env) + '_H' + str(flags.hparams_seed) + '_T' + str(flags.trial_seed)

    return job_id + '.json'

def check_file_integrity(results_dir):

    with open(os.path.join(results_dir,'sweep_config.json'), 'r') as fp:
        flags = json.load(fp)
    
    # Recall sweep config
    flags = Namespace(**flags)
    _, train_args = hparams_sweep.get_train_args(flags)

    # Check for sweep output files
    missing_files = 0
    for args in tqdm.tqdm(train_args, desc="Checking file integrity"):
        name = get_job_json(Namespace(**args))
        
        if not os.path.exists(os.path.join(results_dir, name)):
            missing_files += 1

    assert missing_files == 0, str(missing_files) + " sweep results are missing from the results directory"

def setup_pretty_table(flags, hparams, dataset):

    job_id = 'Training ' + flags.objective  + ' on ' + flags.dataset + ' (H=' + str(flags.hparams_seed) + ', T=' + str(flags.trial_seed) + ')'

    t = PrettyTable()

    env_name = dataset.get_envs()

    if dataset.get_setup() == 'seq':
        t.field_names = ['Env'] + [str(e) if i != flags.test_env else '** ' + str(e) + ' **' for i, e in enumerate(env_name)] + [' ', '  ', '   ']
    if dataset.get_setup() == 'step':
        t.field_names = ['Env'] + [str(e) if i != flags.test_step else '** ' + str(e) + ' **' for i, e in enumerate(env_name)] + [' ', '  ', '   ']

    max_width = {}
    min_width = {}
    for n in t.field_names:
        max_width.update({n: 15})
        min_width.update({n: 15})
    t._min_width = min_width
    t._max_width = max_width
    t.add_row(['Steps'] + ['in   :: out' for e in env_name] + ['Avg Loss', 'Epoch', 'Step Time'])
    print(t.get_string(title=job_id, border=True, hrule=0))
    t.del_row(0)
    
    return t

def get_latex_table(table, caption=None, label=None):
    """Construct and export a LaTeX table from a PrettyTable.
    latexTableExporter(table,**kwargs)
    Inspired from : https://github.com/adasilva/prettytable
    Required argument:
    -----------------
    table - an instance of prettytable.PrettyTable
    Optional keyword arguments:
    --------------------------
    caption - string - a caption for the table
    label - string - the latex reference ID
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
