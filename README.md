# STLD
The code of "Rethinking Self-training for Semi-supervised Landmark Detection: A Selection-free Approach".

## Installation
1. Install Python3 and PyTorch >= v1.13
2. Clone this repository.
```Shell
git clone https://github.com/jhb86253817/STLD.git
```
3. Install the dependencies in requirements.txt.
```Shell
pip install -r requirements.txt
```

## Datasets
Download the datasets from official sources, then put them under folder `data/DATA_NAME/`.
* [300W](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
* [WFLW](https://wywu.github.io/projects/LAB/WFLW.html)
* [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)

## Training
1. Go to `lib`, run `python preprocess.py data_300W` to prepare 300W.
2. Configure the command in `run_train.sh`, then run `bash run_train.sh` to start training.
