
# Downloading the data
Before running any training run, we need to make sure we have the data to train on. 
## Direct Preprocessed Download
The repository offers direct download to the preprocessed data which is the quickest and most efficient way to get started. To download the preprocessed data, run the download module of the woods.scripts package and specify the dataset you want to download:
```sh
python3 -m woods.scripts.download DATASET\
        --data_path ./path/to/data/directory
```
## Source Download and Preprocess
For the sake of transparency, WOODS also offers the preprocessing scripts we took for all datasets in the preprecessing module of the woods.scripts package. You can also use the same module to download the raw data from the original source and run preprocessing yourself on it. DISCLAIMER: Some of the datasets take a long time to preprocess, especially the EEG datasets.
```sh
python3 -m woods.scripts.preprocess DATASET\
        --data_path ./path/to/data/directory
```
## Datasets Info
The following table lists the available datasets and their corresponding raw and preprocessed sizes.

|      Datasets     | Modality  | Requires Download | Preprocessed Size | Raw Size |
|-------------------|-----------|--------------------|-------------------|-------------------|
| Basic_Fourier | 1D Signal | No | - | - | - |
| Spurious_Fourier | 1D Signal | No | - | - | - |
| TMNIST | Video | Yes, but done automatically | 0.11 GB | - |
| TCMNIST_seq | Video | Yes, but done automatically | 0.11 GB | - |
| TCMNSIT_step | Video | Yes, but done automatically | 0.11 GB | - |
| CAP_DB | EEG | Yes | 9.1 GB | 40.1 GB |
| SEDFx_DB | EEG | Yes | 10.7 GB | 8.1 GB |
| LSA64 | Video | Yes | 0.26 GB | 1.5 GB |
| HAR | Sensor | Yes | 0.16 GB | 3.1 GB |
