# Demo package
In our paper "Inferring the temperature from planar velocity measurements in Rayleigh-Bénard convection by deep learning", we describe several deep learning experiments. This repository contains the code to reproduce the final results of the paper. To execute the package you need to process the following steps:

## Installation of requirements
```bash
pip install -r requirements.txt
```
## Get data
```bash
# Currently you need to request the data via email: theo.kaeufer@tu-ilmenau.de
# After receiving the data, you need to put it into the data folder
```

## Run experiments
```bash
python main.py --scenario <scenario>
```
where `<scenario>` is P1 or P2.
```
