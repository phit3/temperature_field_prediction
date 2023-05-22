# Demo package
In our paper "Inferring the temperature from planar velocity measurements in Rayleigh-Bénard convection by deep learning", we describe several deep learning experiments. This repository contains the code to reproduce the final results of the paper. To execute the package you need to process the following steps:

## Installation of requirements
```bash
pip3 install -r requirements.txt
```
## Get data
```bash
mkdir data
# You can request data access via email: theo.kaeufer@tu-ilmenau.de
# After receiving the data, you need to put it into the data folder
```

## Run experiments
```bash
python main.py [--scenario <scenario>] [--repetitions <repetitions>] [--tag <tag>]
```
where `<scenario>` is P0, P1 or P2 (Default: P0).
      `<repetitions>` is the number of repetitions of the experiment (Default: 5).
      `<tag>` is an optional tag to identify the experiment (Default: some UUID).
```
