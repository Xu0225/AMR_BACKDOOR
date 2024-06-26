# Backdoor Attack and Defense Evaluation Methods for Electromagnetic Signal Modulation Recognition
## Environment Setup
```bash
conda env create --name amr_backdoor python=3.8
conda activate amr_backdoor
pip install -r requirements.txt
```

## Data Preparation
### Signal Source Dataset
https://github.com/radioML/dataset
### Constellation Map Dataset (Long Duration)
```bash
cd datasets 
python run_generation.py # generate constellation data
```
## RQ1: AMR Performance Evaluation
### Sequence Representation for AMR Model Evaluation
```bash
cd RQ1/Seq
python main.py
cd RQ2/Seq
python run_seq_attack.py # set trigger_types = ["benign"]
```
### Statistical Feature Representation for AMR Model Evaluation
```bash
cd RQ2/Feat
python run_feat_attack.py # set trigger_types = ["benign"]
```
### Constellation Map Representation for AMR Model Evaluation
```bash
cd RQ2/Img
python run_img_attack.py # set trigger_types = ["benign"]
```

## RQ2: AMR Backdoor Attacks
### Sequential Representation for AMR Model Backdoor Attacks
```bash
cd RQ2/Seq
python run_seq_attack.py
```
### Constellation Map Representation for AMR Model Backdoor Attacks
```bash
cd RQ2/Img 
python run_img_attack.py
```
### Feature Representation for AMR Model Backdoor Attacks
```bash
cd RQ2/Feat 
python run_feat_attack.py
```
## RQ3: AMR Backdoor Defense
```bash
cd RQ3 
python run_defense.py
```
## RQ4: Multimodal AMR
```bash
cd RQ4
python run_multi_modal_attack.py
```