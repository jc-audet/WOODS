""" Directly download the preprocessed data """

import os
import shutil
import gdown
import argparse
import subprocess
import zipfile
import academictorrents as at

from woods.download import download_cap, download_sedfx, download_pcl, download_hhar, download_lsa64

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('dataset', nargs='*', type=str, default=['CAP', 'SEDFx', 'PCL', 'HHAR', 'LSA64'])
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    parser.add_argument('--mode', type=str, default='gdrive', choices=['at', 'gdrive'])
    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))

    if 'CAP' in flags.dataset:
        download_cap(flags.data_path, flags.mode)

    if 'SEDFx' in flags.dataset:
        download_sedfx(flags.data_path, flags.mode)
    
    if 'PCL' in flags.dataset:
        download_pcl(flags.data_path, flags.mode)

    if 'HHAR' in flags.dataset:
        download_hhar(flags.data_path, flags.mode)

    if 'LSA64' in flags.dataset:
        download_lsa64(flags.data_path, flags.mode)
