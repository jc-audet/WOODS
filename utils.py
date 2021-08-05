from prettytable import PrettyTable
import math

def setup_pretty_table(hparams, dataset):

    t = PrettyTable()
    # t.set_style(13)
    env_name = dataset.get_envs()
    step_str = 'Step'
    if len(str(dataset.N_STEPS))+1 > len('steps'):
        step_str = 'Step' + ' '*(len(str(dataset.N_STEPS))+1-len('steps'))

    column_name = []
    for n in env_name:
        if len(str(n)) < 15:
            n = ' '*((15-len('steps')) // 2) + str(n) + ' '*((15-len('steps')) // 2)
        column_name.append(n)
    t.field_names = [step_str] + [e for e in column_name]
    t.add_row([' '] + ['in   :: out' for e in column_name])
    print(t.get_string(title="Hello", border=True, hrule=0))
    t.del_row(0)
    
    return t
