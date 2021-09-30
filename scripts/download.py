import os
import copy
import argparse
import datetime
import numpy as np
import glob
import h5py
import subprocess
import mne
import pyedflib

from lib.datasets import DATASETS

class PhysioNet():
    '''
    PhysioNet Sleep stage dataset
    Download: wget -r -N -c -np https://physionet.org/files/capslpdb/1.0.0/

    TODO:
        * Remove useless data from machine after making the h5 file
        * check if something is already done in the download and if it does, don't do it
        * Make it so we don't need the files attribute with the gather_EEG function
    '''
    files = [
        [   'physionet.org/files/capslpdb/1.0.0/nfle29',
            'physionet.org/files/capslpdb/1.0.0/nfle7',
            'physionet.org/files/capslpdb/1.0.0/nfle1',
            'physionet.org/files/capslpdb/1.0.0/nfle5',
            'physionet.org/files/capslpdb/1.0.0/n11',
            'physionet.org/files/capslpdb/1.0.0/rbd18',
            'physionet.org/files/capslpdb/1.0.0/plm9',
            'physionet.org/files/capslpdb/1.0.0/nfle35',
            'physionet.org/files/capslpdb/1.0.0/nfle36',
            'physionet.org/files/capslpdb/1.0.0/nfle2',
            'physionet.org/files/capslpdb/1.0.0/nfle38',
            'physionet.org/files/capslpdb/1.0.0/nfle39',
            'physionet.org/files/capslpdb/1.0.0/nfle21'],
        [   'physionet.org/files/capslpdb/1.0.0/nfle10',
            'physionet.org/files/capslpdb/1.0.0/nfle11',
            'physionet.org/files/capslpdb/1.0.0/nfle19',
            'physionet.org/files/capslpdb/1.0.0/nfle26',
            'physionet.org/files/capslpdb/1.0.0/nfle23'],
        [   'physionet.org/files/capslpdb/1.0.0/rbd8',
            'physionet.org/files/capslpdb/1.0.0/rbd5',
            'physionet.org/files/capslpdb/1.0.0/rbd11',
            'physionet.org/files/capslpdb/1.0.0/ins8',
            'physionet.org/files/capslpdb/1.0.0/rbd10'],
        [   'physionet.org/files/capslpdb/1.0.0/n3',
            'physionet.org/files/capslpdb/1.0.0/nfle30',
            'physionet.org/files/capslpdb/1.0.0/nfle13',
            'physionet.org/files/capslpdb/1.0.0/nfle18',
            'physionet.org/files/capslpdb/1.0.0/nfle24',
            'physionet.org/files/capslpdb/1.0.0/nfle4',
            'physionet.org/files/capslpdb/1.0.0/nfle14',
            'physionet.org/files/capslpdb/1.0.0/nfle22',
            'physionet.org/files/capslpdb/1.0.0/n5',
            'physionet.org/files/capslpdb/1.0.0/nfle37'],
        [   'physionet.org/files/capslpdb/1.0.0/nfle3',
            'physionet.org/files/capslpdb/1.0.0/nfle40',
            'physionet.org/files/capslpdb/1.0.0/nfle15',
            'physionet.org/files/capslpdb/1.0.0/nfle12',
            'physionet.org/files/capslpdb/1.0.0/nfle28',
            'physionet.org/files/capslpdb/1.0.0/nfle34',
            'physionet.org/files/capslpdb/1.0.0/nfle16',
            'physionet.org/files/capslpdb/1.0.0/nfle17']
    ]

    def __init__(self, flags):
        super(PhysioNet, self).__init__()

        ## Download 
        download_process = subprocess.Popen(['wget', '-r', '-N', '-c', '-np', 'https://physionet.org/files/capslpdb/1.0.0/', '-P', flags.data_path])
        download_process.wait()
        
        ## Process data into machines
        common_channels = self.gather_EEG(flags)

        ## Cluster data into machines and save
        for i, env_set in enumerate(self.files):

            for j, recording in enumerate(env_set):

                # Create data path
                edf_path = os.path.join(flags.data_path, recording + '.edf')
                txt_path = os.path.join(flags.data_path, recording + '.txt')

                # Fetch all data
                data = mne.io.read_raw_edf(edf_path)
                ch = [og_ch for og_ch in data.ch_names if og_ch.lower() in common_channels]
                data = data.pick_channels(ch)
                labels, times = self.read_annotation(txt_path)

                # Get labels
                labels = self.string_2_label(labels)

                # Sample and filter
                data.resample(100)
                data.filter(l_freq=0.3, h_freq=30)

                # Get the indexes
                start = data.info['meas_date']
                times = [(t_s.replace(tzinfo=start.tzinfo), t_e.replace(tzinfo=start.tzinfo))  for (t_s, t_e) in times]
                time_diff = [ ((t_s - start).total_seconds(), (t_e - start).total_seconds()) for (t_s, t_e) in times]
                t_s, t_e = [t_s for (t_s, t_e) in time_diff], [t_e for (t_s, t_e) in time_diff]
                index_s = data.time_as_index(t_s)
                index_e = data.time_as_index(t_e)

                # Split the data 
                seq = np.array([data.get_data(start=s, stop=e) for s, e in zip(index_s, index_e) if e <= len(data)])
                labels = np.array([[l] for l, e in zip(labels, index_e) if e <= len(data)])

                # Add data to container
                env_data = np.zeros((0, 19, 3000))
                env_labels = np.zeros((0, 1))
                env_data = np.append(env_data, seq, axis=0)
                env_labels = np.append(env_labels, labels, axis=0)


                # Reshape and scale the data
                sc = mne.decoding.Scaler(scalings='mean')
                env_data = sc.fit_transform(env_data)
                env_data = np.transpose(env_data, (0,2,1))

                with h5py.File(os.path.join(flags.data_path, 'physionet.org/data.h5'), 'a') as hf:
                    if j == 0:
                        g = hf.create_group('Machine' + str(i))
                        g.create_dataset('data', data=env_data.astype('float32'), dtype='float32', maxshape=(None, 3000, 19))
                        g.create_dataset('labels', data=env_labels.astype('float32'), dtype='int_', maxshape=(None,1))
                    else:
                        hf['Machine' + str(i)]['data'].resize((hf['Machine' + str(i)]['data'].shape[0] + env_data.shape[0]), axis = 0)
                        hf['Machine' + str(i)]['data'][-env_data.shape[0]:,:,:] = env_data
                        hf['Machine' + str(i)]['labels'].resize((hf['Machine' + str(i)]['labels'].shape[0] + env_labels.shape[0]), axis = 0)
                        hf['Machine' + str(i)]['labels'][-env_labels.shape[0]:,:] = env_labels
        
        # Remove useless files
        self.remove_useless(flags)

    def remove_useless(self, flags):

        for file in glob.glob(os.path.join(flags.data_path, 'physionet.org/files/capslpdb/1.0.0/')):
            print("Removing: ", file)
            # os.remove(file)

    def string_2_label(self, string):
        
        label_dict = {  'W':0,
                        'S1':1,
                        'S2':2,
                        'S3':3,
                        'S4':4,
                        'R':5}
                        
        labels = [label_dict[s] for s in string]

        return labels

    def read_annotation(self, txt_path):

        # Initialize storage
        labels = []
        times = []
        durations = []

        with open(txt_path, 'r') as file:
            lines = file.readlines()

        in_table = False
        for line in lines:
            if line[0:16] == 'Recording Date:	':
                date = [int(u) for u in line.strip('\n').split('\t')[1].split('/')]

            if in_table:
                line_list = line.split("\t")
                if line_list[event_id][0:5] == 'SLEEP' and (position_id == None or line_list[position_id] != 'N/A'):
                    labels.append(line_list[label_id])
                    durations.append(line_list[duration_id])
                    t = line_list[time_id].split(':') if ':' in line_list[time_id] else line_list[time_id].split('.')
                    t = [int(u) for u in t]
                    dt = datetime.datetime(*date[::-1], *t) + datetime.timedelta(days=int(t[0]<12))
                    times.append((dt, dt + datetime.timedelta(seconds=int(line_list[duration_id]))))

            if line[0:11] == 'Sleep Stage':
                columns = line.split("\t")
                label_id = columns.index('Sleep Stage')
                time_id = columns.index('Time [hh:mm:ss]')
                duration_id = columns.index('Duration[s]')
                try:
                    position_id = columns.index('Position')
                except ValueError:
                    position_id = None
                event_id = columns.index('Event')
                in_table = True

        return labels, times

    def gather_EEG(self, flags):

        machine_id = 0
        machines = {}
        edf_file = []
        table = []
        for file in glob.glob(os.path.join(flags.data_path, 'physionet.org/files/capslpdb/1.0.0/*.edf')):

            # Fetch all data from file
            edf_file.append(file)
            try:
                data = pyedflib.EdfReader(file)
            except OSError:
                print("Crashed")
                continue

            ch_freq = data.getSampleFrequencies()
            data = mne.io.read_raw_edf(file)
            ch = [c.lower() for c in data.ch_names]

            # Create state Dict (ID)
            state_dict = {}
            for n, f in zip(ch, ch_freq):
                state_dict[n] = f
            state_set = set(state_dict.items())

            # Create or assign ID
            if state_set not in table:
                id = copy.deepcopy(machine_id)
                machine_id +=1
                table.append(state_set)
            else:
                id = table.index(state_set)

            # Add of update the dictionnary
            if id not in machines.keys():
                machines[id] = {}
                machines[id]['state'] = state_set
                machines[id]['amount'] = 1
                machines[id]['dates'] = [data.info['meas_date']]
                machines[id]['names'] = [file]
            else:
                machines[id]['amount'] += 1 
                machines[id]['dates'].append(data.info['meas_date'])
                machines[id]['names'].append(file)
            
        _table = []
        for id, machine in machines.items():
            if machine['amount'] > 4:
                ch = [c[0] for c in machine['state']]
                freq = [c[1] for c in machine['state']]

                _table.append(set(ch))
                print("___________________________________________________")
                print("Machine ID: ", id)
                print("Recording amount: ", machine['amount'])
                print("Channels: ", ch)
                print('Freqs: ', freq)
                print("Dates:")
                for d in machine['dates']:
                    print(d)
                print("Files:")
                for f in machine['names']:
                    print(f)

        return list(set.intersection(*_table))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('dataset', nargs='*', type=str, default=DATASETS)
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))

    if 'PhysioNet' in flags.dataset:
        physionet = PhysioNet(flags)