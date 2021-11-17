"""This module is used to run yourself the raw download and preprocessing of the data

You can directly download the preprocessed data with the download.py module. 
This module is used only for transparancy of how the datasets are preprocessed.
It also gives the opportunity to the most curageous to change the preprocessing approaches of the data for curiosity.

Note:
    The intention of releasing the benchmarks of woods is to investigate the performance of domain generalization techniques.
    Although some preprocessing tricks could lead to better OoD performance, this approach is not encouraged when using the WOODS benchmarks.
"""

import os
import csv
import mne
import copy
import json
import glob
import h5py
import xlrd
import argparse
import datetime
import numpy as np
import subprocess
import pyedflib

# Local import
from woods.datasets import DATASETS

# Preprocessing tools imports
from scipy.signal import resample, detrend
from sklearn.preprocessing import scale

# Torch import
import torchvision
from torchvision.transforms import Compose, Resize, Lambda
from torchvision.transforms._transforms_video import (
    ToTensorVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import UniformTemporalSubsample

# For MI dataset
from moabb.datasets import BNCI2014001, PhysionetMI, Lee2019_MI
from moabb.paradigms import MotorImagery
from moabb import utils

import matplotlib.pyplot as plt

class CAP():
    """ Fetch the data from the PhysioNet website and preprocess it 

    The download is automatic but if you want to manually download::

        wget -r -N -c -np https://physionet.org/files/capslpdb/1.0.0/

    Args:
        flags (argparse.Namespace): The flags of the script
    """
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
        super(CAP, self).__init__()

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

                # Detrend, scale and reshape the data
                sc = mne.decoding.Scaler(scalings='mean')
                env_data = detrend(env_data, axis=2) # detrending
                env_data = sc.fit_transform(env_data) # Normalizing
                env_data = np.transpose(env_data, (0,2,1))

                # Save the data
                preprocessed_path = os.path.join(flags.data_path, 'CAP')
                os.makedirs(preprocessed_path, exist_ok=True)
                with h5py.File(os.path.join(preprocessed_path, 'CAP.h5'), 'a') as hf:
                    if j == 0:
                        g = hf.create_group('Machine' + str(i))
                        g.create_dataset('data', data=env_data.astype('float32'), dtype='float32', maxshape=(None, 3000, 19))
                        g.create_dataset('labels', data=env_labels.astype('float32'), dtype='int_', maxshape=(None,1))
                    else:
                        hf['Machine' + str(i)]['data'].resize((hf['Machine' + str(i)]['data'].shape[0] + env_data.shape[0]), axis = 0)
                        hf['Machine' + str(i)]['data'][-env_data.shape[0]:,:,:] = env_data
                        hf['Machine' + str(i)]['labels'].resize((hf['Machine' + str(i)]['labels'].shape[0] + env_labels.shape[0]), axis = 0)
                        hf['Machine' + str(i)]['labels'][-env_labels.shape[0]:,:] = env_labels
        
        ## Remove useless files
        self.remove_useless(flags)

    def remove_useless(self, flags):
        """ Remove useless files """

        for file in glob.glob(os.path.join(flags.data_path, 'physionet.org/files/capslpdb/1.0.0/*')):
            print("Removing: ", file)
            os.remove(file)
        print("Removing Folder: ", os.path.join(flags.data_path, 'physionet.org/files/capslpdb/1.0.0'))
        os.rmdir(os.path.join(flags.data_path, 'physionet.org/files/capslpdb/1.0.0'))
        print("Removing Folder: ", os.path.join(flags.data_path, 'physionet.org/files/capslpdb'))
        os.rmdir(os.path.join(flags.data_path, 'physionet.org/files/capslpdb'))
        print("Removing Folder: ", os.path.join(flags.data_path, 'physionet.org/files'))
        os.rmdir(os.path.join(flags.data_path, 'physionet.org/files'))
        print("Removing: ", os.path.join(flags.data_path, 'physionet.org/robots.txt'))
        os.remove(os.path.join(flags.data_path, 'physionet.org/robots.txt'))

    def string_2_label(self, string):
        """ Convert string to label """
        
        label_dict = {  'W':0,
                        'S1':1,
                        'S2':2,
                        'S3':3,
                        'S4':4,
                        'R':5}
                        
        labels = [label_dict[s] for s in string]

        return labels

    def read_annotation(self, txt_path):
        """ Read annotation file for the CAP dataset"""

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
        """ Gets the intersection of common channels across all machines 
        
        Returns:
            list: list of channels (strings)
        """
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
    
class SEDFx():
    """ Fetch the PhysioNet Sleep-EDF Database Expanded Dataset and preprocess it
    
    The download is automatic but if you want to manually download::

        wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
        
    Args:
        flags (argparse.Namespace): The flags of the script
    """

    def __init__(self, flags):
        super(SEDFx, self).__init__()

        ## Download 
        download_process = subprocess.Popen(['wget', '-r', '-N', '-c', '-np', 'https://physionet.org/files/sleep-edfx/1.0.0/', '-P', flags.data_path])
        download_process.wait()

        ## Process data into machines
        common_channels = self.gather_EEG(flags)

        ## Set labels
        label_dict = {  'Sleep stage W':0,
                        'Sleep stage 1':1,
                        'Sleep stage 2':2,
                        'Sleep stage 3':3,
                        'Sleep stage 4':4,
                        'Sleep stage R':5}

        ## Get subjects from xls file
        SC_dict = {}
        SC_xls = xlrd.open_workbook(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/SC-subjects.xls')).sheet_by_index(0)
        for row in range(1, SC_xls.nrows):
            if int(SC_xls.cell_value(row,0)) not in SC_dict.keys():
                SC_dict[int(SC_xls.cell_value(row,0))] = {}
                SC_dict[int(SC_xls.cell_value(row,0))]['nights'] = ['SC4{:02d}{}'.format(int(SC_xls.cell_value(row,0)), int(SC_xls.cell_value(row,1)))]
                SC_dict[int(SC_xls.cell_value(row,0))]['folder'] = 'physionet.org/files/sleep-edfx/1.0.0/sleep-cassette'
            else:
                SC_dict[int(SC_xls.cell_value(row,0))]['nights'].append('SC4{:02d}{}'.format(int(SC_xls.cell_value(row,0)), int(SC_xls.cell_value(row,1))))
            SC_dict[int(SC_xls.cell_value(row,0))]['age'] = int(SC_xls.cell_value(row,2))
            SC_dict[int(SC_xls.cell_value(row,0))]['sex'] = int(SC_xls.cell_value(row,3))
        ST_dict = {}
        ST_xls = xlrd.open_workbook(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/ST-subjects.xls')).sheet_by_index(0)
        for row in range(2, ST_xls.nrows):
            ST_dict[int(ST_xls.cell_value(row,0))] = {}
            ST_dict[int(ST_xls.cell_value(row,0))]['folder'] = 'physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry'
            ST_dict[int(ST_xls.cell_value(row,0))]['nights'] = ['ST7{:02d}{}'.format(int(ST_xls.cell_value(row,0)), int(ST_xls.cell_value(row,3))), 
                                                                'ST7{:02d}{}'.format(int(ST_xls.cell_value(row,0)), int(ST_xls.cell_value(row,5)))]
            ST_dict[int(ST_xls.cell_value(row,0))]['age'] = int(ST_xls.cell_value(row,1))
            ST_dict[int(ST_xls.cell_value(row,0))]['sex'] = 2 if int(ST_xls.cell_value(row,2))==1 else 1

        ## Create group in h5 file
        dummy_data = np.zeros((0,3000,4))
        dummy_labels = np.zeros((0,1))
        groups = ['Age 20-40', 'Age 40-60', 'Age 60-80', 'Age 80-100']
        preprocessed_path = os.path.join(flags.data_path, 'SEDFx')
        os.makedirs(preprocessed_path, exist_ok=True)
        with h5py.File(os.path.join(preprocessed_path, 'SEDFx.h5'), 'a') as hf:
            for g in groups:
                g = hf.create_group(g)
                g.create_dataset('data', data=dummy_data.astype('float32'), dtype='float32', maxshape=(None, 3000, 4))
                g.create_dataset('labels', data=dummy_labels.astype('float32'), dtype='int_', maxshape=(None,1))

        ## Cluster data into machines and save
        for db in [SC_dict, ST_dict]:
            for subject, subject_info in db.items():

                # Find Age group
                if 20 < subject_info['age'] <= 40:
                    age_group = groups[0]
                elif 40 < subject_info['age'] <= 60:
                    age_group = groups[1]
                elif 60 < subject_info['age'] <= 80:
                    age_group = groups[2]
                elif 80 < subject_info['age']:
                    age_group = groups[3]
                else:
                    print("Age group counldn't be found")
                
                for night in subject_info['nights']:
                    edf_path = os.path.join(flags.data_path, subject_info['folder'], night+ '*')

                    # Fetch file name
                    PSG_file = glob.glob(edf_path+'PSG.edf')[0]
                    hypno_file = glob.glob(edf_path+'Hypnogram.edf')[0]

                    # Read raw data and pick channels
                    data = mne.io.read_raw_edf(PSG_file)
                    ch = [og_ch for og_ch in data.ch_names if og_ch.lower() in common_channels]
                    data = data.pick_channels(ch)
                    data.resample(100)
                    data.filter(l_freq=0.3, h_freq=30)

                    # Get annotations i.e. labels, crop the big start and end chunks of recordings
                    annot = mne.read_annotations(hypno_file)
                    annot.crop(annot[1]['onset'] - 30 * 60, annot[-2]['onset'] + 30 * 60)
                    data.set_annotations(annot, emit_warning=False)

                    events, event_id = mne.events_from_annotations(data, chunk_duration=30., event_id=label_dict)
                    # mne.viz.plot_events(events, sfreq=data.info['sfreq'])
                    tmax = 30. - 1. / data.info['sfreq']  # tmax in included

                    epochs_data = mne.Epochs(raw=data, events=events,
                                            event_id=event_id, tmin=0., tmax=tmax, baseline=None)
                    
                    # Add data to container
                    input_data = epochs_data.get_data()
                    labels = events[:,2:]

                    # Reshape and scale the data
                    sc = mne.decoding.Scaler(scalings='mean')
                    input_data = detrend(input_data, axis=2) # detrending
                    input_data = sc.fit_transform(input_data) # Normalizing
                    input_data = np.transpose(input_data, (0,2,1))
                    
                    with h5py.File(os.path.join(preprocessed_path, 'SEDFx.h5'), 'a') as hf:
                        hf[age_group]['data'].resize((hf[age_group]['data'].shape[0] + input_data.shape[0]), axis = 0)
                        hf[age_group]['data'][-input_data.shape[0]:,:,:] = input_data
                        hf[age_group]['labels'].resize((hf[age_group]['labels'].shape[0] + labels.shape[0]), axis = 0)
                        hf[age_group]['labels'][-labels.shape[0]:,:] = labels

        # Remove useless files
        self.remove_useless(flags)

    def remove_useless(self, flags):
        """ Remove useless files """

        for file in glob.glob(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/*')):
            print("Removing: ", file)
            os.remove(file)
        print("Removing Folder: ", os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry'))
        os.rmdir(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry'))
        for file in glob.glob(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/*')):
            print("Removing: ", file)
            os.remove(file)
        print("Removing Folder: ", os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/sleep-cassette'))
        os.rmdir(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/sleep-cassette'))
        for file in glob.glob(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/*')):
            print("Removing: ", file)
            os.remove(file)
        print("Removing Folder: ", os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0'))
        os.rmdir(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0'))
        print("Removing Folder: ", os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx'))
        os.rmdir(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx'))
        print("Removing Folder: ", os.path.join(flags.data_path, 'physionet.org/files'))
        os.rmdir(os.path.join(flags.data_path, 'physionet.org/files'))
        print("Removing: ", os.path.join(flags.data_path, 'physionet.org/robots.txt'))
        os.remove(os.path.join(flags.data_path, 'physionet.org/robots.txt'))

    def string_2_label(self, string):
        """ Convert string to label """
        
        label_dict = {  'W':0,
                        'S1':1,
                        'S2':2,
                        'S3':3,
                        'S4':4,
                        'R':5}
                        
        labels = [label_dict[s] for s in string]

        return labels

    def read_annotation(self, txt_path):
        """ Read annotation file """

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
        """ Gets the intersection of common channels across all machines 
        
        Returns:
            list: list of channels (strings)
        """

        machine_id = 0
        machines = {}
        edf_file = []
        table = []
        for file in glob.glob(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/*PSG.edf')):

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

        for file in glob.glob(os.path.join(flags.data_path, 'physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/*PSG.edf')):

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

def StockVolatility(flags):

    # ###
    # Need to Create 5 environments (2000-2004, 2005-2009, 2010-2014, 2015-2019)
    # Steps:
    # Our data is already cleaned and in sequence, we can transform the CSV into a dictionary
    # ###
    envs = ['2000-2004', '2005-2009', '2010-2014', '2015-2020']
    data = [[], [], [], []]
    labels = [[], [], [], []]
    #This code opens every group of 5 years of volatility from many stock indices,
    #and places the volatilities in the appropriate lists.
    for f in range(0, 4):
        rownumber = 0

        i = f * 5
        file_name = "StockVolatility/%sworks.csv" % (str(2000 + i) + "-" + str(2000 + i + 4))
        file_path = os.path.join(flags.data_path, file_name)
        with open(file_path, encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if rownumber % 4 == 3:
                    labels[f].append(row['Volatility'])
                else:
                    data[f].append(row['Volatility'])
                rownumber += 1

    #Now, we can transform the data into chunks for 3 timesteps into np arrays.
    chunked_data = [[], [], [], []]
    time_steps = 3
    chunked_labels = [[], [], [], []]
    for indx, val in enumerate(data):
        for i in range(0, len(data[indx]), time_steps):
            chunked_data[indx].append(data[indx][i:i + time_steps])
    for indx, val in enumerate(labels):

        for i in range(0, len(labels[indx]), 1):
            chunked_labels[indx].append(data[indx][i:i + 1])

    for indx, val in enumerate(chunked_data):
        chunked_data[indx] = np.expand_dims(np.array(chunked_data[indx]), axis=-1)

    for indx, val in enumerate(chunked_labels):
        chunked_labels[indx] = np.array(chunked_labels[indx])

    with h5py.File(os.path.join(flags.data_path, 'StockVolatility/StockVolatility.h5'), 'a') as hf:
        #Make a new environment for each index of the np arrays.
        for index,env in enumerate(envs):
            if env in hf.keys():
                del hf[env]
            g = hf.create_group(env)
            g.create_dataset('data', data=chunked_data[index].astype('float32'), dtype='float32')
            g.create_dataset('labels', data=chunked_labels[index].astype('float32'), dtype='float32')

def HAR(flags):
    """ Fetch and preprocess the HAR dataset

    Note:
        You need to manually download the HAR dataset from the source and place it in the data folder in order to preprocess it yourself:

            https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition

    Args:
        flags (argparse.Namespace): The flags of the script
    """
    # Label definition
    label_dict = {  'stand': 0,
                    'sit': 1,
                    'walk': 2,
                    'bike': 3,
                    'stairsup': 4,
                    'stairsdown': 5,
                    'null': 6}

    ## Fetch all data and put it all in a big dict
    data_dict = {}
    for file in glob.glob(os.path.join(flags.data_path, 'HAR/*.csv')):
        print(file)

        # Get modality
        if 'gyroscope' in file:
            mod = 'gyro'
        elif 'accelerometer' in file:
            mod = 'acc'

        # Get number of time steps for all recordings
        with open(file) as f:
            data = csv.reader(f)
            next(data)
            for row in data:
                if row[8] not in data_dict.keys():
                    print(row[8])
                    data_dict[row[8]] = {}
                if row[6] not in data_dict[row[8]].keys():
                    print('\t' + row[6])
                    data_dict[row[8]][row[6]] = {}
                if mod not in data_dict[row[8]][row[6]].keys():
                    print('\t\t' + mod)
                    data_dict[row[8]][row[6]][mod] = {}
                    data_dict[row[8]][row[6]][mod]['n_pt'] = 0
                
                data_dict[row[8]][row[6]][mod]['n_pt'] += 1

        # Get data
        with open(file) as f:
            data = csv.reader(f)
            next(data)
            for row in data:
                if 'index' not in data_dict[row[8]][row[6]][mod].keys():
                    i = 0
                    data_dict[row[8]][row[6]][mod]['index'] = np.zeros((data_dict[row[8]][row[6]][mod]['n_pt']))
                    data_dict[row[8]][row[6]][mod]['time'] = np.zeros((data_dict[row[8]][row[6]][mod]['n_pt']))
                    data_dict[row[8]][row[6]][mod]['meas'] = np.zeros((data_dict[row[8]][row[6]][mod]['n_pt'],3), dtype=np.float64)
                    data_dict[row[8]][row[6]][mod]['label'] = np.zeros((data_dict[row[8]][row[6]][mod]['n_pt']))
                
                data_dict[row[8]][row[6]][mod]['index'][i] = int(row[0])
                data_dict[row[8]][row[6]][mod]['time'][i] = float(row[2]) / 1e6 # Convert to miliseconds
                data_dict[row[8]][row[6]][mod]['meas'][i,:] = [float(row[3]), float(row[4]), float(row[5])]
                data_dict[row[8]][row[6]][mod]['label'][i] = int(label_dict[row[9]])

                i += 1

    # Delete keys that either 
    # - is missing one modality (e.g. all sansungold devices only have one modality for some reason)or 
    # - has a number of datapoint that is too low (e.g. gear_2 -> 'i' only has 1 point for some reason)
    to_delete = []
    for device in data_dict.keys():
        for sub in data_dict[device].keys():
            if len(data_dict[device][sub].keys()) != 2:
                print("....")
                print("len")
                print(device, sub)
                to_delete.append((device, sub))
                continue
            for mod in data_dict[device][sub].keys():
                if data_dict[device][sub][mod]['n_pt'] < 10000:
                    print("....")
                    print("n_pt")
                    print(data_dict[device][sub][mod]['n_pt'])
                    print(device, sub)
                    to_delete.append((device, sub))
                    break
    for key in to_delete:
        del data_dict[key[0]][key[1]]
    print(to_delete)

    ## Sort data
    for device in data_dict.keys():
        for sub in data_dict[device].keys():
            for mod in data_dict[device][sub].keys():
                # Sort by index
                index_sort = np.argsort(data_dict[device][sub][mod]['index'])
                data_dict[device][sub][mod]['index'] = np.take_along_axis(data_dict[device][sub][mod]['index'], index_sort, axis=0)
                data_dict[device][sub][mod]['time'] = np.take_along_axis(data_dict[device][sub][mod]['time'], index_sort, axis=0)
                data_dict[device][sub][mod]['meas'] = data_dict[device][sub][mod]['meas'][index_sort,:]
                data_dict[device][sub][mod]['label'] = np.take_along_axis(data_dict[device][sub][mod]['label'], index_sort, axis=0)

                # This is to take data that is within recording time 
                # (To see an example of somewhere this isn't the case, check phones_gyrscope -> nexus4_1 -> a -> index [24641, 24675])
                inliers = np.argwhere(  np.logical_and( data_dict[device][sub][mod]['time'][0] <= data_dict[device][sub][mod]['time'], 
                                                        data_dict[device][sub][mod]['time'] <= data_dict[device][sub][mod]['time'][-1]))[:,0]
                
                # Sort by time value
                time_sort = np.argsort(data_dict[device][sub][mod]['time'][inliers])

                data_dict[device][sub][mod]['index'] = data_dict[device][sub][mod]['index'][inliers][time_sort]
                data_dict[device][sub][mod]['time'] = data_dict[device][sub][mod]['time'][inliers][time_sort]
                data_dict[device][sub][mod]['meas'] = data_dict[device][sub][mod]['meas'][inliers][time_sort,:]
                data_dict[device][sub][mod]['label'] = data_dict[device][sub][mod]['label'][inliers][time_sort]

    device_env_mapping = {  'nexus4_1': 'nexus4',
                            'nexus4_2': 'nexus4',
                            's3_1': 's3',
                            's3_2': 's3',
                            's3mini_1': 's3mini',
                            's3mini_2': 's3mini',
                            'gear_1': 'gear',
                            'gear_2': 'gear',
                            'lgwatch_1': 'lgwatch',
                            'lgwatch_2': 'lgwatch'}

    for device in data_dict.keys():
        for i, sub in enumerate(data_dict[device].keys()):
            print("..........")
            print(device, sub)
            # print(len(data_dict[device][sub]['gyro']['time']), data_dict[device][sub]['gyro']['time'][0], data_dict[device][sub]['gyro']['time'][-1])
            # print(len(data_dict[device][sub]['acc']['time']), data_dict[device][sub]['acc']['time'][0], data_dict[device][sub]['acc']['time'][-1])

            tmin = np.max([data_dict[device][sub]['gyro']['time'][0], data_dict[device][sub]['acc']['time'][0]])
            tmax = np.min([data_dict[device][sub]['gyro']['time'][-1], data_dict[device][sub]['acc']['time'][-1]])
            # print(tmin, tmax)

            gyro_in = np.argwhere(  np.logical_and( tmin <= data_dict[device][sub]['gyro']['time'], 
                                                    data_dict[device][sub]['gyro']['time'] <= tmax))[:,0]
            acc_in = np.argwhere(  np.logical_and( tmin <= data_dict[device][sub]['acc']['time'], 
                                                    data_dict[device][sub]['acc']['time'] <= tmax))[:,0]

            data_dict[device][sub]['gyro']['index'] = data_dict[device][sub]['gyro']['index'][gyro_in]
            data_dict[device][sub]['gyro']['time'] = data_dict[device][sub]['gyro']['time'][gyro_in]
            data_dict[device][sub]['gyro']['meas'] = data_dict[device][sub]['gyro']['meas'][gyro_in]
            data_dict[device][sub]['gyro']['label'] = data_dict[device][sub]['gyro']['label'][gyro_in]
            data_dict[device][sub]['acc']['index'] = data_dict[device][sub]['acc']['index'][acc_in]
            data_dict[device][sub]['acc']['time'] = data_dict[device][sub]['acc']['time'][acc_in]
            data_dict[device][sub]['acc']['meas'] = data_dict[device][sub]['acc']['meas'][acc_in]
            data_dict[device][sub]['acc']['label'] = data_dict[device][sub]['acc']['label'][acc_in]

            gyro_in = np.argwhere(data_dict[device][sub]['gyro']['label'] != 6)[:,0]
            acc_in = np.argwhere(data_dict[device][sub]['acc']['label'] != 6)[:,0]

            data_dict[device][sub]['gyro']['index'] = data_dict[device][sub]['gyro']['index'][gyro_in]
            data_dict[device][sub]['gyro']['time'] = data_dict[device][sub]['gyro']['time'][gyro_in]
            data_dict[device][sub]['gyro']['meas'] = data_dict[device][sub]['gyro']['meas'][gyro_in,:]
            data_dict[device][sub]['gyro']['label'] = data_dict[device][sub]['gyro']['label'][gyro_in]
            data_dict[device][sub]['acc']['index'] = data_dict[device][sub]['acc']['index'][acc_in]
            data_dict[device][sub]['acc']['time'] = data_dict[device][sub]['acc']['time'][acc_in]
            data_dict[device][sub]['acc']['meas'] = data_dict[device][sub]['acc']['meas'][acc_in,:]
            data_dict[device][sub]['acc']['label'] = data_dict[device][sub]['acc']['label'][acc_in]

            ## Scale data
            data_dict[device][sub]['gyro']['meas'] = scale(data_dict[device][sub]['gyro']['meas'])
            data_dict[device][sub]['acc']['meas'] = scale(data_dict[device][sub]['acc']['meas'])

            # Resample and split the data here
            idx = 0
            data = np.zeros((0,500,6))
            labels = np.zeros((0,1))
            while True:
                if idx >= len(data_dict[device][sub]['gyro']['time'])-1:
                    break
                start_time = data_dict[device][sub]['gyro']['time'][idx]
                gyro_in = np.argwhere(  np.logical_and( start_time <= data_dict[device][sub]['gyro']['time'], 
                                                        data_dict[device][sub]['gyro']['time'] <= start_time+5000))[:,0]
                acc_in = np.argwhere(  np.logical_and( start_time <= data_dict[device][sub]['acc']['time'],
                                                        data_dict[device][sub]['acc']['time'] <= start_time+5000))[:,0]
                print(len(gyro_in), len(acc_in))
                                        
                if len(gyro_in) == 0 or len(acc_in) == 0:
                    # print("time not intersecting segment")
                    idx += len(gyro_in)
                    continue       
                if data_dict[device][sub]['gyro']['time'][gyro_in][-1] - data_dict[device][sub]['gyro']['time'][gyro_in][0] < 4900 or data_dict[device][sub]['acc']['time'][acc_in][-1] - data_dict[device][sub]['acc']['time'][acc_in][0] < 4900:
                    # print("end on break segment")
                    idx += len(gyro_in)
                    continue
                if len(np.argwhere(np.diff(data_dict[device][sub]['gyro']['time'][gyro_in]) > 200)[:,0]) > 0 :
                    diff = np.argwhere(np.diff(data_dict[device][sub]['gyro']['time'][gyro_in]) > 200)[:,0]
                    # print("gyro contains a break")
                    idx += diff[-1]+1
                    continue
                if len(np.argwhere(np.diff(data_dict[device][sub]['acc']['time'][acc_in]) > 200)[:,0]) > 0:
                    diff = np.argwhere(np.diff(data_dict[device][sub]['acc']['time'][acc_in]) > 200)[:,0]
                    # print("acc contains a break")
                    idx += diff[-1]+1
                    continue
                start_label = data_dict[device][sub]['gyro']['label'][idx]
                if len(np.argwhere(data_dict[device][sub]['gyro']['label'][gyro_in] != start_label)[:,0]) > 0:
                    labels_diff = np.argwhere(data_dict[device][sub]['gyro']['label'][gyro_in] != start_label)[:,0]
                    # print("label switch in sequence")
                    idx += labels_diff[0]+1
                    continue
                
                idx += len(gyro_in)

                time = np.linspace(start = data_dict[device][sub]['gyro']['time'][gyro_in][0], stop=data_dict[device][sub]['gyro']['time'][gyro_in][-1], num=500)
                gyro_dat = resample(data_dict[device][sub]['gyro']['meas'][gyro_in, :], 500)
                acc_dat = resample(data_dict[device][sub]['acc']['meas'][acc_in, :], 500)

                all_dat = np.concatenate((acc_dat, gyro_dat), axis=1)
                data = np.concatenate((data, np.expand_dims(all_dat, axis=0)), axis=0)
                labels = np.concatenate((labels, np.expand_dims(data_dict[device][sub]['gyro']['label'][gyro_in][0:1], axis=0)), axis=0)
            
            env = device_env_mapping[device]

            with h5py.File(os.path.join(flags.data_path, 'HAR/HAR.h5'), 'a') as hf:
                if env not in hf.keys():
                    g = hf.create_group(env)
                    g.create_dataset('data', data=data.astype('float32'), dtype='float32', maxshape=(None, 500, 6))
                    g.create_dataset('labels', data=labels.astype('float32'), dtype='int_', maxshape=(None,1))
                else:
                    hf[env]['data'].resize((hf[env]['data'].shape[0] + data.shape[0]), axis = 0)
                    hf[env]['data'][-data.shape[0]:,:,:] = data
                    hf[env]['labels'].resize((hf[env]['labels'].shape[0] + labels.shape[0]), axis = 0)
                    hf[env]['labels'][-labels.shape[0]:,:] = labels

def LSA64(flags):
    """ Fetch the LSA64 dataset and preprocess it

    Note:
        You need to manually download the HAR dataset from the source and place it in the data folder in order to preprocess it yourself:

            https://mega.nz/file/FQJGCYba#uJKGKLW1VlpCpLCrGVu89wyQnm9b4sKquCOEAjW5zMo

    Args:
        flags (argparse.Namespace): The flags of the script
    """

    for person in range(1,11):
        person_ID = str(person).zfill(3)
        
        for i, file in enumerate(glob.glob(os.path.join(flags.data_path, 'LSA64', '*_'+person_ID+'_*'))):
            print(str(i+1)+ ' / 320 (' + file+')')
            ID = file.split('/')[-1].split('_')
            sample_num = ID[-1].split('.')[0]
                
            vid = torchvision.io.read_video(os.path.join(flags.data_path, 'LSA64', file), end_pts=2.5, pts_unit='sec')[0]

            transform = Compose([ToTensorVideo(),
                                 Resize(size=(224, 224)),
                                 UniformTemporalSubsample(20)])#,
                                #  NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                #                 std=[0.229, 0.224, 0.225])])
            vid = transform(vid)

            if not os.path.exists(os.path.join(flags.data_path, 'LSA64', ID[1])):
                os.makedirs(os.path.join(flags.data_path, 'LSA64', ID[1]))
            if not os.path.exists(os.path.join(flags.data_path, 'LSA64', ID[1], ID[0])):
                os.makedirs(os.path.join(flags.data_path, 'LSA64', ID[1], ID[0]))
            if not os.path.exists(os.path.join(flags.data_path, 'LSA64', ID[1], ID[0], sample_num)):
                os.mkdir(os.path.join(flags.data_path, 'LSA64', ID[1], ID[0], sample_num))
            for frame in range(vid.shape[1]):
                torchvision.utils.save_image(vid[:,frame,...], os.path.join(flags.data_path, 'LSA64', ID[1], ID[0], sample_num, 'frame_'+str(frame).zfill(6)+'.jpg'))

class MI():
    '''
    This class helps to download and prepare MI datasets
    
    '''

    def __init__(self,flags):
        super(MI, self).__init__()

        self.path = flags.data_path

        print('Downloading MI datasets')
        mne.set_config('MNE_DATASETS_BNCI_PATH', self.path)
        utils.set_download_dir(self.path)

        # Datasets
        ds_src1 = PhysionetMI()
        ds_src2 = BNCI2014001()
        ds_src3 = Lee2019_MI()

        #find common channels and freq. filtering
        fmin, fmax = 4, 32
        raw = ds_src2.get_data(subjects=[1])[1]['session_T']['run_1']
        src2_channels = raw.pick_types(eeg=True).ch_names
        raw = ds_src3.get_data(subjects=[1])[1]['session_2']['train']
        src3_channels = raw.pick_types(eeg=True).ch_names
        common_channels = set(src2_channels) & set(src3_channels)
        sfreq = 250.
        prgm_2classes = MotorImagery(n_classes=2, channels=common_channels, resample=sfreq, fmin=fmin, fmax=fmax)
        prgm_4classes = MotorImagery(n_classes=4, channels=common_channels, resample=sfreq, fmin=fmin, fmax=fmax)

        print("Fetching data")
        X_src1, label_src1, m_src1 = prgm_4classes.get_data(dataset=ds_src1, subjects=list(range(1,110)))  
        print("First source dataset has {} trials with {} electrodes and {} time samples".format(*X_src1.shape))
        print ("Source dataset 1 include labels: {}".format(np.unique(label_src1)))
        X_src2, label_src2, m_src2 = prgm_4classes.get_data(dataset=ds_src2, subjects=list(range(1,10)))  
        print("Second source dataset has {} trials with {} electrodes and {} time samples".format(*X_src2.shape))
        print ("Source dataset 2 include labels: {}".format(np.unique(label_src2)))
        X_src3, label_src3, m_src3 = prgm_2classes.get_data(dataset=ds_src3, subjects=list(range(1,40)))  
        print("Third source dataset has {} trials with {} electrodes and {} time samples".format(*X_src3.shape))
        print ("Source dataset 3 include labels: {}".format(np.unique(label_src3)))

        y_src1 = np.array([self.relabel(l) for l in label_src1])
        y_src2 = np.array([self.relabel(l) for l in label_src2])
        y_src3 = np.array([self.relabel(l) for l in label_src3])

        print("Only right-/left-hand labels are used and first source dataset does not have other labels:")
        print(np.unique(y_src1), np.unique(y_src2), np.unique(y_src3))     

        # Deleting trials of "other labels"
        print("Deleting trials from 'other labels'")
        X_src1 = np.delete(X_src1,y_src1==2,0)
        y_src1 = np.delete(y_src1,y_src1==2,0)
        X_src2 = np.delete(X_src2,y_src2==2,0)
        y_src2 = np.delete(y_src2,y_src2==2,0)

        ## windowing trails
        window_size = min(X_src1.shape[2], X_src2.shape[2], X_src3.shape[2])
        X_src1 = X_src1[:, :, :window_size]
        X_src2 = X_src2[:, :, :window_size]
        X_src3 = X_src3[:, :, :window_size]

        ## Create group in h5 file
        dummy_data = np.zeros((0,window_size,len(common_channels)))
        dummy_labels = np.zeros((0,1))
        groups = ['PhysionetMI', 'BNCI2014001', 'Lee2019_MI']
        X = [X_src1, X_src2, X_src3]
        Y = [y_src1, y_src2, y_src3]
        with h5py.File(os.path.join(self.path, 'MI/MI.h5'), 'a') as hf:
            for g in groups:
                g = hf.create_group(g)
                g.create_dataset('data', data=dummy_data.astype('float32'), dtype='float32', maxshape=(None, window_size, len(common_channels)))
                g.create_dataset('labels', data=dummy_labels.astype('float32'), dtype='int_', maxshape=(None,1))
        
        ## Save data to h5 file
        for group, x, y in zip(groups,X,Y):
            with h5py.File(os.path.join(self.path, 'MI/MI.h5'), 'a') as hf:
                hf[group]['data'].resize((hf[group]['data'].shape[0] + x.shape[0]), axis = 0)
                hf[group]['data'][-x.shape[0]:,:,:] = x.transpose((0,2,1))
                hf[group]['labels'].resize((hf[group]['labels'].shape[0] + y.shape[0]), axis = 0)
                hf[group]['labels'][-y.shape[0]:,:] = y.reshape([-1,1])
    
    def relabel(self,l):
        """ Converts labels from str to int """
        if l == 'left_hand': return 0
        elif l == 'right_hand': return 1
        else: return 2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('dataset', nargs='*', type=str, default=DATASETS)
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))

    if 'CAP' in flags.dataset:
        CAP(flags)

    if 'SEDFx' in flags.dataset:
        SEDFx(flags)
    
    if 'MI' in flags.dataset:
        MI(flags)

    if 'StockVolatility' in flags.dataset:
        StockVolatility(flags)

    if 'HAR' in flags.dataset:
        HAR(flags)

    if 'LSA64' in flags.dataset:
        LSA64(flags)
