from prettytable import PrettyTable
import math

def setup_pretty_table(hparams, dataset):

    t = PrettyTable()
    t.set_style(13)
    env_name = dataset.get_envs()
    step_str = 'Step'
    if len(str(hparams['steps'])) > len('steps'):
        step_str = 'Step' + ' '*(len(str(training_hparams['steps']))-len('steps'))

    column_name = []
    for n in env_name:
        if len(str(n)) < 13:
            n = ' '*((13-len('steps')) // 2) + str(n) + ' '*((13-len('steps')) // 2)
        column_name.append(n)
    t.field_names = [step_str] + [e for e in column_name]
    print(t)
    
    return t
