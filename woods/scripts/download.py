""" Directly download the preprocessed data """
import os
import gdown
import requests
import argparse
import subprocess

# Local import
from woods.datasets import DATASETS


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_URL(url, path):
    """ Download a file from a URL and save it to a path """
    
    archive_file = os.path.join(path, "files.zip")
    gdown.cached_download(url=url, path=archive_file, quiet=False, postprocess=gdown.extractall)
    os.remove(path=archive_file)

def CAP(data_path):
    """ Download the CAP dataset """

    url = "https://drive.google.com/uc?id=1NFwX2CqLrenWF4az0c6J-OglAoD48PAT"
    path = os.path.join(data_path, "CAP")
    os.makedirs(name=path, exist_ok=True)

    download_URL(url, path)

def SEDFx(data_path):
    """ Download the SEDFx dataset """

    url = "https://drive.google.com/uc?id=15j_bsiOmMJb42mG712Vhv3jZ4MQSOgoT"
    path = os.path.join(data_path, "SEDFx")
    os.makedirs(name=path, exist_ok=True)

    download_URL(url, path)

def PCL(data_path):
    """ Download the PCL dataset """

    url = "https://drive.google.com/uc?id=118DNxHpzeJwVTM22wzZhSiOuDsno0nay"
    path = os.path.join(data_path, "PCL")
    os.makedirs(name=path, exist_ok=True)

    download_URL(url, path)

def HHAR(data_path):
    """ Download the HHAR dataset """

    url = "https://drive.google.com/uc?id=1Z3IcrCE-o77p4YrvkCy-Y-0CgCyxVHet"
    path = os.path.join(data_path, "HHAR")
    os.makedirs(name=path, exist_ok=True)

    download_URL(url, path)

def LSA64(data_path):
    """ Download the LSA64 dataset """

    url = "https://drive.google.com/uc?id=1YwwSg8Dt178ySp3ht_BLJwl5j5s_IU1m"
    path = os.path.join(data_path, "LSA64")
    os.makedirs(name=path, exist_ok=True)

    download_URL(url, path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('dataset', nargs='*', type=str, default=DATASETS)
    parser.add_argument('--data_path', type=str, default='~/Documents/Data/')
    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))

    if 'CAP' in flags.dataset:
        CAP(flags.data_path)

    if 'SEDFx' in flags.dataset:
        SEDFx(flags.data_path)
    
    if 'PCL' in flags.dataset:
        PCL(flags.data_path)

    if 'HHAR' in flags.dataset:
        HHAR(flags.data_path)

    if 'LSA64' in flags.dataset:
        LSA64(flags.data_path)
