
# Quick Installation

WOODS is still under active developpement so it is still only available by cloning the repository on your local machine. 

## Installing requirements

### With Conda

First, have conda installed on your machine (see their [installation page](https://docs.anaconda.com/anaconda/install/) if that is not the case). Then create a conda environment with the following command:
```sh
conda create --name woods python=3.9
```
Then activate the environment with the following command:
```sh
conda activate woods
```
### With venv
You can use the python virtual environment manager [virtualenv](https://virtualenv.pypa.io/en/latest/) to create a virtual environment for the project.
```sh
python3 -m venv /path/to/env/woods
```
Then activate the virtual environment with the following command:
```sh
source /path/to/env/woods/bin/activate
```

## Clone locally
Once you've created the virtual environment, clone the repository. 
```sh
git clone something something
cd woods
```
Then install the requirements with the following command:
```sh
pip install -r requirements.txt
```

## Run tests
Run the tests to make sure everything is in order. 
```sh
Coming soon!
```