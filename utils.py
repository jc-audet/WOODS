from prettytable import PrettyTable
import math

def setup_pretty_table(flags, hparams, dataset):

    job_id = 'Training ' + flags.objective  + flags.dataset + ' (H=' + str(flags.hparams_seed) + ', T=' + str(flags.trial_seed) + ')'

    t = PrettyTable()

    env_name = dataset.get_envs()

    t.field_names = ['Env'] + [e for e in env_name]

    max_width = {'Env': 5}
    min_width = {'Env': 5}
    for n in env_name:
        max_width.update({str(n): 15})
        min_width.update({str(n): 15})
    t._min_width = min_width
    t._max_width = max_width
    t.add_row(['Steps'] + ['in   :: out' for e in env_name])
    print(t.get_string(title=job_id, border=True, hrule=0))
    t.del_row(0)
    
    return t
