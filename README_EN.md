<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:06B6D4,45:6366F1,100:8B5CF6&height=220&section=header&text=AMR%20BACKDOOR&fontSize=52&fontColor=FFFFFF&fontAlignY=35&desc=Backdoor%20Attack%20%26%20Defense%20for%20Deep-Learning-based%20Automatic%20Modulation%20Recognition&descAlignY=55&descSize=16&animation=fadeIn"/>

# 📡 AMR BACKDOOR

### Backdoor Attack & Defense Evaluation for Deep-Learning-based Automatic Modulation Recognition

<p>
A research-oriented experimental framework for studying the <b>security of Automatic Modulation Recognition (AMR)</b> systems under backdoor attacks.
</p>

<p>
Sequence · Statistical Features · Constellation Maps · Multimodal Fusion
</p>

<p>
<img src="https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Dataset-RadioML%202016.10A-00B4D8"/>
<img src="https://img.shields.io/badge/Task-Automatic%20Modulation%20Recognition-6366F1"/>
<img src="https://img.shields.io/badge/Security-Backdoor%20Learning-DC2626"/>
<img src="https://img.shields.io/github/stars/Xu0225/AMR_BACKDOOR?style=flat&logo=github"/>
</p>

<br/>

**Clean Signal → Trigger Injection → Multi-view Representation → AMR Model → CA / ASR → Defense**

</div>

---

## ⚡ Overview

Deep-learning-based **Automatic Modulation Recognition (AMR)** has demonstrated strong performance in intelligent wireless communication systems. However, its vulnerability to **training-time backdoor attacks** remains an important security concern.

**AMR_BACKDOOR** provides an experimental framework for systematically investigating backdoor behavior across multiple signal representations, AMR architectures, trigger designs, and defense mechanisms.

The repository is organized around four research questions:

| Research Question | Focus |
|---|---|
| **RQ1** | How well do representative deep AMR models perform under clean conditions? |
| **RQ2** | How vulnerable are AMR models to backdoor attacks across different signal representations? |
| **RQ3** | How effective are existing backdoor defenses for AMR? |
| **RQ4** | What happens when AMR moves from a single representation to multimodal recognition? |

---

## 🔥 What Makes This Repository Different?

<table>
<tr>
<td width="50%" valign="top">

### 📡 Multi-view Signal Representation

The same wireless signal is studied through multiple views:

- **IQ** — raw in-phase / quadrature samples
- **AP** — amplitude / phase representation
- **FFT** — frequency-domain representation
- **Statistical features**
- **Expert features / higher-order statistics**
- **Constellation maps**
- **Multimodal fusion**

</td>
<td width="50%" valign="top">

### 🧨 Backdoor Trigger Zoo

Implemented trigger configurations include:

- BadNet
- Random Location
- Hanning
- Spectrum Shift
- Phase Shift
- Remapped AWGN
- Benign baseline

Different triggers can be evaluated under the same AMR pipeline.

</td>
</tr>

<tr>
<td width="50%" valign="top">

### 🧠 Model Zoo

**Sequence models**

CNN2 · VTCNN2 · CLDNN · GRU · LSTM · CGDNN · MCLDNN · ResNet · DenseNet · ICAMC · MCNET · DAE ...

**Feature models**

CART · XGBoost · LightGBM

**Image models**

CNN · VGG16 · ResNet50

</td>
<td width="50%" valign="top">

### 🛡️ Defense Benchmark

The defense modules include:

- STRIP
- Spectral Signatures
- Activation Clustering
- Fine-Pruning
- t-SNE visualization

This enables attack and defense experiments inside the same AMR setting.

</td>
</tr>
</table>

---

## 🧬 Experimental Pipeline

~~~mermaid
flowchart LR
    A["📡 RadioML 2016.10A<br/>I/Q Signals"]

    A --> B1["〰️ Sequence View<br/>IQ · AP · FFT"]
    A --> B2["📊 Feature View<br/>Basic · Time · Expert"]
    A --> B3["🟣 Constellation Map"]

    B1 --> C["🧨 Backdoor Trigger Injection"]
    B2 --> C
    B3 --> C

    C --> D1["🧠 Deep AMR Models"]
    C --> D2["🌳 Feature-based ML"]
    C --> D3["🖼️ Vision Models"]

    D1 --> E["📈 Evaluation"]
    D2 --> E
    D3 --> E

    E --> F1["Clean Accuracy<br/>CA"]
    E --> F2["Attack Success Rate<br/>ASR"]
    E --> F3["Per-SNR Accuracy"]

    D1 --> G["🛡️ Defense"]
    G --> G1["STRIP"]
    G --> G2["Spectral Signatures"]
    G --> G3["Activation Clustering"]
    G --> G4["Fine-Pruning"]

    B1 --> H["🔀 Multimodal AMR"]
    B2 --> H
    B3 --> H
~~~

---

# 🧪 Research Questions

## RQ1 — Clean AMR Performance

RQ1 establishes the benign performance baseline before introducing backdoor triggers.

A collection of representative deep AMR architectures is evaluated on **RadioML 2016.10A**, including convolutional, recurrent, hybrid, and residual architectures.

~~~bash
cd RQ1
python main.py
~~~

The evaluation reports recognition accuracy across different **Signal-to-Noise Ratio (SNR)** conditions.

Experimental artifacts are also included in:

~~~text
RQ1/
├── RQ1_results.xlsx
├── RQ1_results.pptx
├── results/
└── saved_model/
~~~

---

## RQ2 — Backdoor Attacks

RQ2 studies whether the vulnerability of AMR systems changes with the **representation of the electromagnetic signal**.

### ① Sequence Representation

Supported representations:

~~~text
IQ
AP
FFT
~~~

Default experimental models:

~~~text
CNN2
CLDNNLikeModel
GRUModel
~~~

Run:

~~~bash
cd RQ2/Seq
python run_seq_attack.py
~~~

The runner evaluates combinations of:

~~~text
representation × model × trigger
~~~

with both **Clean Accuracy (CA)** and **Attack Success Rate (ASR)**.

---

### ② Statistical / Expert Feature Representation

Three feature views are included:

~~~text
basic
time
expert
~~~

Models:

~~~text
CART
XGBoost
LightGBM
~~~

Run:

~~~bash
cd RQ2/Feat
python run_attack.py
~~~

The feature pipeline contains notebooks for feature extraction and analysis, including:

~~~text
Feature_Extraction.ipynb
amr_feature_extraction.ipynb
CNN_FEAT.ipynb
CART_XGB_GBDT.ipynb
~~~

---

### ③ Constellation Map Representation

AMR is also evaluated as an image-recognition problem using constellation diagrams.

Models:

~~~text
CNN
VGG16
ResNet50
~~~

Run:

~~~bash
cd RQ2/Img
python run_img_attack.py
~~~

This pipeline evaluates whether backdoor behaviors transfer from raw wireless signal representations to **visual signal representations**.

---

# 🛡️ RQ3 — Backdoor Defense

The repository contains several representative backdoor detection / mitigation methods:

~~~text
RQ3/
├── AC/       # Activation Clustering
├── FP/       # Fine-Pruning
├── SS/       # Spectral Signatures
├── STRIP/    # STRIP
├── TSNE/     # Representation visualization
└── results/
~~~

Example Fine-Pruning experiment:

~~~bash
cd RQ3

python ./FP/fine_pruning.py     --TRIGGER_TYPE badnet     --POS_RATE 0.1     --EPOCH 100     --MODEL_NAME CNN2     --REP IQ
~~~

Other defense implementations can be executed directly from their corresponding directories.

> **Note**
>
> <code>run_defense.py</code> is a legacy experiment driver. In its current version, several command assignments are overwritten sequentially, so the final command is the one that is actually executed.
> For reproducible experiments, running each defense module explicitly is recommended.

---

# 🔀 RQ4 — Multimodal AMR

RQ4 explores backdoor behavior when AMR no longer relies on a single representation.

The repository contains experiments for:

~~~text
Single-model inference
Ensemble inference
Stacking-based multimodal fusion
~~~

Relevant files:

~~~text
RQ4/
├── main.py
├── main_ensemble.py
├── main_stacking.py
├── main_test.py
└── run_multi_modal_attack.py
~~~

Run the configured multimodal experiments with:

~~~bash
cd RQ4
python run_multi_modal_attack.py
~~~

The current experiment runner uses the **stacking-based multimodal pipeline**.

---

# 🧨 Trigger Configuration

The central trigger implementation is located in:

~~~text
trigger_config.py
~~~

Attack configurations currently include:

| Trigger | Idea |
|---|---|
| **BadNet** | Fixed local trigger injected into selected samples |
| **Random Location** | Trigger injected at varying temporal positions |
| **Hanning** | Applies a Hanning-window-based signal transformation |
| **Spectrum Shift** | Manipulates the signal in the frequency domain |
| **Phase Shift** | Introduces a phase-domain perturbation |
| **Remapped AWGN** | Uses remapped noise as the trigger pattern |
| **Benign** | No malicious trigger; used as the reference baseline |

The default poisoning rate used by the experiment runners is typically **10%**.

---

# 📊 Evaluation Metrics

Two primary metrics are used throughout the backdoor experiments.

### Clean Accuracy — CA

Recognition accuracy on benign samples after training the potentially poisoned model.

> High CA → the backdoored model still behaves normally on clean inputs.

### Attack Success Rate — ASR

Recognition success toward the attacker's target label after the trigger is present.

> High ASR → the implanted backdoor is highly effective.

The framework additionally records performance across different **SNR levels**, making it possible to study the interaction between:

~~~text
Channel Quality × Signal Representation × Model × Trigger
~~~

---

# 📂 Repository Structure

~~~text
AMR_BACKDOOR/
│
├── RQ1/
│   ├── main.py
│   ├── results/
│   ├── saved_model/
│   ├── RQ1_results.xlsx
│   └── RQ1_results.pptx
│
├── RQ2/
│   ├── Seq/                 # Sequence-based AMR attacks
│   │   ├── main.py
│   │   └── run_seq_attack.py
│   │
│   ├── Feat/                # Feature-based AMR attacks
│   │   ├── main.py
│   │   ├── run_attack.py
│   │   └── *.ipynb
│   │
│   └── Img/                 # Constellation-map attacks
│       ├── img_bd_attack.py
│       └── run_img_attack.py
│
├── RQ3/
│   ├── AC/
│   ├── FP/
│   ├── SS/
│   ├── STRIP/
│   ├── TSNE/
│   └── run_defense.py
│
├── RQ4/
│   ├── main.py
│   ├── main_ensemble.py
│   ├── main_stacking.py
│   └── run_multi_modal_attack.py
│
├── datasets/
│   ├── generate_img.py
│   └── run_generation.py
│
├── rmlmodel/
│   ├── Sequence/
│   └── Image/
│
├── mltools.py
├── trigger_config.py
└── README.md
~~~

---

# 🚀 Getting Started

## 1. Clone

~~~bash
git clone https://github.com/Xu0225/AMR_BACKDOOR.git
cd AMR_BACKDOOR
~~~

## 2. Create the Environment

The original project was developed around **Python 3.8**.

~~~bash
conda create -n amr_backdoor python=3.8 -y
conda activate amr_backdoor
~~~

The source code uses packages including:

~~~text
TensorFlow / Keras
NumPy
Pandas
SciPy
scikit-learn
XGBoost
LightGBM
Matplotlib
openpyxl
~~~

> Dependency versions were not frozen in the original repository.
> Reconstructing a TensorFlow/Keras-compatible Python 3.8 environment is recommended for reproduction.

---

# 📡 Dataset

The experiments are based primarily on:

### RadioML 2016.10A

Original dataset project:

https://github.com/radioML/dataset

After downloading the dataset, configure the path used by:

~~~text
trigger_config.py
RQ1/main.py
~~~

---

## Generate Constellation Maps

~~~bash
cd datasets
python run_generation.py
~~~

The generated constellation representations are used by the image-based AMR experiments.

---

# ⚙️ Reproducibility Notes

This repository preserves the original experimental code and therefore contains several **legacy machine-specific paths**, for example:

~~~text
D:/zhaixu/Thesis_Code/
~~~

and

~~~text
/root/zx/Thesis_Code/
~~~

Before reproducing experiments on a new machine, search the repository for these paths and replace them with the local project / dataset directories.

In particular, check:

~~~text
trigger_config.py
RQ1/main.py
RQ2/Seq/main.py
RQ2/Feat/main.py
RQ2/Img/img_bd_attack.py
~~~

A future cleanup can replace these absolute paths with a centralized configuration such as <code>config.yaml</code>, <code>.env</code>, or CLI arguments to make the project fully portable.

---

# 🧠 Research Scope

~~~mermaid
mindmap
  root((AMR Security))
    Wireless Signal Processing
      I/Q Signals
      SNR
      Spectrum
      Constellation
    Machine Learning
      CNN
      RNN / GRU / LSTM
      Tree Models
      Multimodal Learning
    AI Security
      Data Poisoning
      Backdoor Attacks
      Trigger Design
      Backdoor Detection
      Model Repair
~~~

---

# 🔬 Why AMR Backdoors Matter

AMR models can become part of intelligent spectrum monitoring, cognitive radio, electromagnetic environment perception, and autonomous communication systems.

A compromised AMR model may:

- behave normally during conventional testing,
- maintain high recognition accuracy for benign signals,
- but systematically misclassify signals containing a hidden trigger.

This makes backdoors particularly difficult to identify using ordinary accuracy-based validation alone.

The project therefore evaluates both:

> **“Does the model still look normal?” — CA**

and

> **“Does the hidden behavior activate?” — ASR**

---

# ⚠️ Research & Responsible Use

This repository is intended for:

- academic research,
- AI security evaluation,
- AMR robustness studies,
- backdoor defense research,
- and reproducible experimentation.

Experiments should be performed only on datasets, models, and systems that you are authorized to evaluate.

---

<div align="center">

## ⭐ If this repository helps your research

Consider giving it a star — it helps others discover the project.

<br/>

**Built for exploring the boundary between wireless intelligence and trustworthy AI.**

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:8B5CF6,50:6366F1,100:06B6D4&height=120&section=footer"/>

</div>
