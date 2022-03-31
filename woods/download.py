import os
import shutil
import argparse
import zipfile
import academictorrents as at
import gdown


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

def download_cap(data_path, mode):
    """ Download the CAP dataset """

    if mode == 'gdrive':

        # Create the path
        path = os.path.join(data_path, "cap")
        os.makedirs(name=path, exist_ok=True)

        # Get the data
        archive_url = "https://drive.google.com/uc?id=1NFwX2CqLrenWF4az0c6J-OglAoD48PAT"
        download_gdrive(archive_url, path, 'cap.zip')

    elif mode == 'at':

        # Get the data
        at_hash = '500d0c473108ef72e01b0f8037251b09331467f9'
        download_at(at_hash, data_path, 'cap.zip')

    # Rename directories and data
    shutil.move(os.path.join(data_path, "cap"), os.path.join(data_path, "CAP"))

def download_sedfx(data_path, mode):
    """ Download the SEDFx dataset """

    if mode == 'gdrive':

        # Create the path
        path = os.path.join(data_path, "sedfx")
        os.makedirs(name=path, exist_ok=True)

        # Get the data
        archive_url = "https://drive.google.com/uc?id=15j_bsiOmMJb42mG712Vhv3jZ4MQSOgoT"
        download_gdrive(archive_url, path, 'sedfx.zip')

    elif mode == 'at':

        # Get the data
        at_hash = '58ea303dce39ffe822bec7704f9eb65e4173defd'
        download_at(at_hash, data_path, 'sedfx.zip')

    # Rename directories and data
    shutil.move(os.path.join(data_path, "sedfx"), os.path.join(data_path, "SEDFx"))

def download_pcl(data_path, mode):
    """ Download the PCL dataset """

    if mode == 'gdrive':

        # Create the path
        path = os.path.join(data_path, "pcl")
        os.makedirs(name=path, exist_ok=True)

        # Get the data
        archive_url = "https://drive.google.com/uc?id=118DNxHpzeJwVTM22wzZhSiOuDsno0nay"
        download_gdrive(archive_url, path, 'pcl.zip')

    elif mode == 'at':

        # Get the data
        at_hash = 'e8b0a24177988f9c3f8c3c63a8212546f67a25a3'
        download_at(at_hash, data_path, 'pcl.zip')

    # Rename directories and data
    shutil.move(os.path.join(data_path, "pcl"), os.path.join(data_path, "PCL"))
    os.rename(os.path.join(data_path, "PCL", 'MI.h5'), os.path.join(data_path, "PCL", 'PCL.h5'))

def download_hhar(data_path, mode):
    """ Download the HHAR dataset """

    if mode == 'gdrive':

        url = "https://drive.google.com/uc?id=1Z3IcrCE-o77p4YrvkCy-Y-0CgCyxVHet"
        path = os.path.join(data_path, "hhar")
        os.makedirs(name=path, exist_ok=True)

        # Get the data
        archive_url = "https://drive.google.com/uc?id=1Z3IcrCE-o77p4YrvkCy-Y-0CgCyxVHet"
        download_gdrive(archive_url, path, 'hhar.zip')

    elif mode == 'at':

        # Get the data
        at_hash = 'f48f38de06b3cd560fb90307b5a1997a12bcc29c'
        download_at(at_hash, data_path, 'hhar.zip')

    # Rename directories and data
    shutil.move(os.path.join(data_path, "hhar"), os.path.join(data_path, "HHAR"))
    os.rename(os.path.join(data_path, "HHAR", 'HAR.h5'), os.path.join(data_path, "HHAR", 'HHAR.h5'))

def download_lsa64(data_path, mode):
    """ Download the LSA64 dataset """

    if mode == 'gdrive':

        # Create the path
        path = os.path.join(data_path, "lsa64")
        os.makedirs(name=path, exist_ok=True)

        # Get the data
        url = "https://drive.google.com/uc?id=1YwwSg8Dt178ySp3ht_BLJwl5j5s_IU1m"
        download_gdrive(url, path, 'lsa64.zip')

    elif mode == 'at':

        # Get the data
        at_hash = '704bf5981eb337cae7cb518c3abb9d7b6bdf3e49'
        download_at(at_hash, data_path, 'lsa64.zip')

    # Rename directories and data
    shutil.move(os.path.join(data_path, "lsa64"), os.path.join(data_path, "LSA64"))