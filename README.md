# Spiking Radar Gestures
This repository contains the supplementary code for the paper **Hand Gesture Recognition in Range-Doppler Images Using Binary Activated Spiking Neural Networks**.

## Prerequisites
 - Python3
 - Tensorflow > 2.4.1
 - tensorflow-datasets

## Setup
To run the experiments, the two datasets ([deep-soli](https://github.com/simonwsw/deep-soli/blob/master/README.md), [TinyRadarNN](https://tinyradar.ethz.ch/)) need to be downloaded and preprocessed. This is handeled by tensorflow-datasets.

```
cd src/datasets/interfacing_soli_dataset
tfds build

cd src/datasets/tinyradar_dataset
tfds build
```

## Run
The `experiments.py` script is the main file for the simulations. Specify the experiment name and dataset there:

```python
if __name__ == '__main__':
    FLAGS = {
        'batch_size': 256,
        'epochs': 30,
        'experiment_name': 'final_spiking_loo_wta_010', # in src/experiments 
        'dataset': 'tinyradar' # or 'soli'
    }
```