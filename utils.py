from prettytable import PrettyTable
import math

def setup_pretty_table(flags, hparams, dataset):

    job_id = 'Training ' + flags.objective + ' on ' + flags.dataset + ' (H=' + str(flags.hparams_seed) + ', T=' + str(flags.trial_seed) + ')'

    t = PrettyTable()

    env_name = dataset.get_envs()
    step_str = 'Env'
    if len(str(dataset.N_STEPS))+1 > len('Env'):
        step_str = ' '*((len(str(dataset.N_STEPS))+1-len('Env'))//2) + 'Env' + ' '*((len(str(dataset.N_STEPS))+1-len('Env'))//2)

    column_name = []
    for n in env_name:
        if len(str(n)) < 15:
            n = ' '*((14-len('Env')) // 2) + str(n) + ' '*((15-len('Env')) // 2)
        column_name.append(n)
    t.field_names = [step_str] + [e for e in column_name]
    t.add_row(['Steps'] + ['in   :: out' for e in column_name])
    print(t.get_string(title=job_id, border=True, hrule=0))
    t.del_row(0)
    
    return t
