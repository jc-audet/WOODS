""" Directly download the preprocessed data """

import os
import shutil
import gdown
import requests
import argparse
import subprocess
import zipfile
import academictorrents as at

# Local import
from woods.datasets import DATASETS

def download_gdrive(url, path, archive_name):
    """ Download the preprocessed data from google drive """
    
    # Download the file
    r = gdown.download(url, os.path.join(path, archive_name), quiet=False)

    # Extract the archive to the path
    with zipfile.ZipFile(r, 'r') as zip_ref:
        zip_ref.extractall(path)

    # Remove the unnecessary archive
    os.remove(r)

def download_at(at_hash, path, archive_name):
    """ Download the preprocessed data from academic torrents """
    
    # Download the torrent
    r = at.get(at_hash, datastore=path, showlogs=True)

    # Extract the archive to the path
    with zipfile.ZipFile(os.path.join(r, archive_name), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(r))
    
    # Remove the unnecessary archive
    os.remove(os.path.join(r, archive_name))

def CAP(data_path, mode):
    """ Download the CAP dataset """

    if mode == 'gdrive':
        url = "https://drive.google.com/uc?id=1NFwX2CqLrenWF4az0c6J-OglAoD48PAT"
        path = os.path.join(data_path, "cap")
        os.makedirs(name=path, exist_ok=True)

        download_gdrive(url, path, 'cap.zip')

    elif mode == 'at':

        at_hash = '500d0c473108ef72e01b0f8037251b09331467f9'
        download_at(at_hash, data_path, 'cap.zip')
    
    # Rename directories and data
    shutil.move(os.path.join(data_path, "cap"), os.path.join(data_path, "CAP"))

def SEDFx(data_path, mode):
    """ Download the SEDFx dataset """

    if mode == 'gdrive':
        url = "https://drive.google.com/uc?id=15j_bsiOmMJb42mG712Vhv3jZ4MQSOgoT"
        path = os.path.join(data_path, "sedfx")
        os.makedirs(name=path, exist_ok=True)

        download_gdrive(url, path, 'sedfx.zip')

    elif mode == 'at':

        at_hash = '58ea303dce39ffe822bec7704f9eb65e4173defd'
        download_at(at_hash, data_path, 'sedfx.zip')

    # Rename directories and data
    shutil.move(os.path.join(data_path, "sedfx"), os.path.join(data_path, "SEDFx"))

def PCL(data_path, mode):
    """ Download the PCL dataset """

    if mode == 'gdrive':
        url = "https://drive.google.com/uc?id=118DNxHpzeJwVTM22wzZhSiOuDsno0nay"
        path = os.path.join(data_path, "pcl")
        os.makedirs(name=path, exist_ok=True)

        download_gdrive(url, path, 'pcl.zip')

    elif mode == 'at':

        at_hash = 'e8b0a24177988f9c3f8c3c63a8212546f67a25a3'
        download_at(at_hash, data_path, 'pcl.zip')

    # Rename directories and data
    shutil.move(os.path.join(data_path, "pcl"), os.path.join(data_path, "PCL"))
    os.rename(os.path.join(data_path, "PCL", 'MI.h5'), os.path.join(data_path, "PCL", 'PCL.h5'))

def HHAR(data_path, mode):
    """ Download the HHAR dataset """

    if mode == 'gdrive':

        url = "https://drive.google.com/uc?id=1Z3IcrCE-o77p4YrvkCy-Y-0CgCyxVHet"
        path = os.path.join(data_path, "hhar")
        os.makedirs(name=path, exist_ok=True)

        download_gdrive(url, path, 'hhar.zip')

    elif mode == 'at':

        at_hash = 'f48f38de06b3cd560fb90307b5a1997a12bcc29c'
        download_at(at_hash, data_path, 'hhar.zip')

    # Rename directories and data
    shutil.move(os.path.join(data_path, "hhar"), os.path.join(data_path, "HHAR"))
    os.rename(os.path.join(data_path, "HHAR", 'HAR.h5'), os.path.join(data_path, "HHAR", 'HHAR.h5'))

def LSA64(data_path, mode):
    """ Download the LSA64 dataset """

    if mode == 'gdrive':

        url = "https://drive.google.com/uc?id=1YwwSg8Dt178ySp3ht_BLJwl5j5s_IU1m"
        path = os.path.join(data_path, "lsa64")
        os.makedirs(name=path, exist_ok=True)

        download_gdrive(url, path, 'lsa64.zip')

    elif mode == 'at':

        at_hash = '704bf5981eb337cae7cb518c3abb9d7b6bdf3e49'
        download_at(at_hash, data_path, 'lsa64.zip')

    # Rename directories and data
    shutil.move(os.path.join(data_path, "lsa64"), os.path.join(data_path, "LSA64"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('dataset', nargs='*', type=str, default=DATASETS)
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    parser.add_argument('--mode', type=str, default='gdrive', choices=['at', 'gdrive'])
    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))

    if 'CAP' in flags.dataset:
        CAP(flags.data_path, flags.mode)

    if 'SEDFx' in flags.dataset:
        SEDFx(flags.data_path, flags.mode)
    
    if 'PCL' in flags.dataset:
        PCL(flags.data_path, flags.mode)

    if 'HHAR' in flags.dataset:
        HHAR(flags.data_path, flags.mode)

    if 'LSA64' in flags.dataset:
        LSA64(flags.data_path, flags.mode)
