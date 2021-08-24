from prettytable import PrettyTable
import math

def get_job_json(flags):

    if flags.sample_hparams:
        job_id = flags.objective + '_' + flags.dataset + '_' + str(flags.test_env) + '_H' + str(flags.hparams_seed) + '_T' + str(flags.trial_seed)
    else:
        job_id = flags.objective + '_' + flags.dataset + '_' + str(flags.test_env)
    job_json = job_id + '.json'

    return job_json

def check_file_integrity(results_dir):

    with open(os.path.join(results_dir,'sweep_config.json'), 'r') as fp:
        flags = json.load(flags_dict, fp)
    
    






def setup_pretty_table(flags, hparams, dataset):

    job_id = 'Training ' + flags.objective  + ' on ' + flags.dataset + ' (H=' + str(flags.hparams_seed) + ', T=' + str(flags.trial_seed) + ')'

    t = PrettyTable()

    env_name = dataset.get_envs()

    t.field_names = ['Env'] + [str(e) if i != flags.test_env else '** ' + str(e) + ' **' for i, e in enumerate(env_name)]

    max_width = {}
    min_width = {}
    for n in t.field_names:
        max_width.update({n: 15})
        min_width.update({n: 15})
    t._min_width = min_width
    t._max_width = max_width
    t.add_row(['Steps'] + ['in   :: out' for e in env_name])
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