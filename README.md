<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:06B6D4,45:6366F1,100:8B5CF6&height=220&section=header&text=AMR%20BACKDOOR&fontSize=52&fontColor=FFFFFF&fontAlignY=35&desc=Backdoor%20Attack%20%26%20Defense%20for%20Automatic%20Modulation%20Recognition&descAlignY=55&descSize=16&animation=fadeIn"/>

# 📡 AMR BACKDOOR

### 面向深度学习自动调制识别的后门攻击与防御评测框架

<p>用于系统研究 <b>Automatic Modulation Recognition（AMR，自动调制识别）</b> 模型在后门攻击场景下安全性的实验型研究仓库。</p>

<p>序列信号 · 统计特征 · 星座图 · 多模态融合</p>

<p>
<img src="https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Dataset-RadioML%202016.10A-00B4D8"/>
<img src="https://img.shields.io/badge/Task-AMR-6366F1"/>
<img src="https://img.shields.io/badge/Security-Backdoor%20Learning-DC2626"/>
<img src="https://img.shields.io/github/stars/Xu0225/AMR_BACKDOOR?style=flat&logo=github"/>
</p>

<br/>

**干净信号 → 触发器注入 → 多视角表征 → AMR 模型 → CA / ASR → 后门防御**

<br/>

**简体中文** · [English](./README_EN.md)

</div>

---

## ⚡ 项目简介

深度学习驱动的 **自动调制识别（AMR）** 已广泛用于智能无线通信、频谱感知与电磁信号分析，但模型在训练阶段可能遭受 **后门攻击（Backdoor Attack）**：攻击者只需向部分训练样本植入特定触发模式，就可能使模型在正常输入下保持高精度，却在触发器出现时稳定产生攻击者指定的错误输出。

**AMR_BACKDOOR** 面向这一问题，构建了一套覆盖多种信号表征、AMR 模型、后门触发器、攻击指标与防御方法的实验框架。

整个仓库围绕四个研究问题组织：

| 研究问题 | 核心内容 |
|---|---|
| **RQ1** | 代表性深度学习 AMR 模型在正常条件下的识别性能如何？ |
| **RQ2** | 不同信号表征下，AMR 模型对后门攻击的脆弱性有何差异？ |
| **RQ3** | 现有后门检测与缓解方法在 AMR 场景中是否有效？ |
| **RQ4** | 从单一表征扩展到多模态 AMR 后，后门攻击行为如何变化？ |

---

## 🔥 项目亮点

<table>
<tr>
<td width="50%" valign="top">

### 📡 多视角信号表征

同一无线信号从不同表示空间进行建模：

- **IQ**：同相 / 正交原始采样
- **AP**：幅度 / 相位表示
- **FFT**：频域表示
- **统计特征**
- **专家特征 / 高阶统计量**
- **星座图**
- **多模态融合**

</td>
<td width="50%" valign="top">

### 🧨 后门触发器集合

目前代码中包含：

- BadNet
- Random Location
- Hanning
- Spectrum Shift
- Phase Shift
- Remapped AWGN
- Benign 基线

便于在同一实验管线下比较不同触发器。

</td>
</tr>

<tr>
<td width="50%" valign="top">

### 🧠 多类 AMR 模型

**序列模型**

CNN2 · VTCNN2 · CLDNN · GRU · LSTM · CGDNN · MCLDNN · ResNet · DenseNet · ICAMC · MCNET · DAE ...

**特征模型**

CART · XGBoost · LightGBM

**图像模型**

CNN · VGG16 · ResNet50

</td>
<td width="50%" valign="top">

### 🛡️ 攻防一体化评测

防御模块包括：

- STRIP
- Spectral Signatures
- Activation Clustering
- Fine-Pruning
- t-SNE 可视化

可在同一 AMR 场景下完成攻击与防御实验。

</td>
</tr>
</table>

---

## 🧬 实验总体流程

~~~mermaid
flowchart LR
    A["📡 RadioML 2016.10A<br/>I/Q 信号"]
    A --> B1["〰️ 序列表征<br/>IQ · AP · FFT"]
    A --> B2["📊 特征表征<br/>Basic · Time · Expert"]
    A --> B3["🟣 星座图表征"]
    B1 --> C["🧨 后门触发器注入"]
    B2 --> C
    B3 --> C
    C --> D1["🧠 深度 AMR 模型"]
    C --> D2["🌳 特征机器学习模型"]
    C --> D3["🖼️ 图像模型"]
    D1 --> E["📈 攻击效果评测"]
    D2 --> E
    D3 --> E
    E --> F1["Clean Accuracy<br/>CA"]
    E --> F2["Attack Success Rate<br/>ASR"]
    E --> F3["不同 SNR 下准确率"]
    D1 --> G["🛡️ 后门防御"]
    G --> G1["STRIP"]
    G --> G2["Spectral Signatures"]
    G --> G3["Activation Clustering"]
    G --> G4["Fine-Pruning"]
    B1 --> H["🔀 多模态 AMR"]
    B2 --> H
    B3 --> H
~~~

---

# 🧪 RQ1：正常条件下的 AMR 性能

RQ1 用于建立后门攻击前的 **Benign Baseline**。仓库对多种典型深度 AMR 架构进行训练与测试，包括卷积、循环、混合网络以及残差网络等。

~~~bash
cd RQ1
python main.py
~~~

评测结果按照不同 **SNR（Signal-to-Noise Ratio，信噪比）** 输出识别准确率。

~~~text
RQ1/
├── RQ1_results.xlsx
├── RQ1_results.pptx
├── results/
└── saved_model/
~~~

---

# 🧨 RQ2：AMR 后门攻击

RQ2 的核心问题是：

> **当同一个无线信号被表示为序列、人工特征或星座图时，模型对后门攻击的脆弱性是否一致？**

## ① 序列表征攻击

支持：

~~~text
IQ
AP
FFT
~~~

默认实验模型：

~~~text
CNN2
CLDNNLikeModel
GRUModel
~~~

运行：

~~~bash
cd RQ2/Seq
python run_seq_attack.py
~~~

该脚本批量遍历：

~~~text
信号表征 × 模型 × 触发器
~~~

并同时计算 **Clean Accuracy（CA）** 与 **Attack Success Rate（ASR）**。

---

## ② 统计 / 专家特征攻击

支持三类特征视角：

~~~text
basic
time
expert
~~~

实验模型：

~~~text
CART
XGBoost
LightGBM
~~~

运行：

~~~bash
cd RQ2/Feat
python run_attack.py
~~~

特征提取与分析相关 Notebook 包括：

~~~text
Feature_Extraction.ipynb
amr_feature_extraction.ipynb
CNN_FEAT.ipynb
CART_XGB_GBDT.ipynb
~~~

---

## ③ 星座图表征攻击

该分支将 AMR 转换为图像识别问题，通过星座图进行模型训练和后门攻击评估。

实验模型：

~~~text
CNN
VGG16
ResNet50
~~~

运行：

~~~bash
cd RQ2/Img
python run_img_attack.py
~~~

该实验用于研究后门行为是否会从原始无线信号表示迁移到 **视觉化信号表示**。

---

# 🛡️ RQ3：AMR 后门防御

仓库实现或集成了多种代表性后门检测 / 缓解方法：

~~~text
RQ3/
├── AC/       # Activation Clustering
├── FP/       # Fine-Pruning
├── SS/       # Spectral Signatures
├── STRIP/    # STRIP
├── TSNE/     # 特征空间可视化
└── results/
~~~

以 Fine-Pruning 为例：

~~~bash
cd RQ3
python ./FP/fine_pruning.py --TRIGGER_TYPE badnet --POS_RATE 0.1 --EPOCH 100 --MODEL_NAME CNN2 --REP IQ
~~~

> **说明**
>
> 当前 **run_defense.py** 属于早期实验驱动脚本，其中多个 command 变量会被连续覆盖，因此实际上只会执行最后一次赋值的命令。复现实验时，更建议直接运行各个防御目录中的具体脚本。

---

# 🔀 RQ4：多模态 AMR

RQ4 进一步研究当 AMR 不再只依赖单一信号表示时，后门攻击的行为变化。

仓库包含：

~~~text
单模型推理
Ensemble 集成
Stacking 多模态融合
~~~

相关代码：

~~~text
RQ4/
├── main.py
├── main_ensemble.py
├── main_stacking.py
├── main_test.py
└── run_multi_modal_attack.py
~~~

运行：

~~~bash
cd RQ4
python run_multi_modal_attack.py
~~~

当前批量实验脚本默认使用 **Stacking 多模态融合流程**。

---

# 🧨 触发器设计

所有核心触发逻辑集中在：

~~~text
trigger_config.py
~~~

| 触发器 | 核心思路 |
|---|---|
| **BadNet** | 在选定样本的固定局部位置注入触发模式 |
| **Random Location** | 在不同样本的随机时间位置注入触发器 |
| **Hanning** | 对信号应用 Hanning 窗变换 |
| **Spectrum Shift** | 在频域中对信号进行频谱扰动 |
| **Phase Shift** | 引入相位域偏移 |
| **Remapped AWGN** | 将重新映射后的噪声作为触发模式 |
| **Benign** | 不植入后门，用于正常基线实验 |

实验运行器中常用的投毒比例为 **10%**。

---

# 📊 评测指标

### Clean Accuracy — CA

对可能已经被投毒训练过的模型，使用正常测试样本计算识别准确率。

> **高 CA**：后门模型在正常输入上仍然表现得像一个“正常模型”。

### Attack Success Rate — ASR

在测试样本植入触发器后，统计模型被诱导到攻击目标类别的成功率。

> **高 ASR**：说明后门触发机制具有较高有效性。

同时，实验还按不同 SNR 输出结果，用于分析：

~~~text
信道质量 × 信号表征 × 模型结构 × 后门触发器
~~~

之间的交互影响。

---

# 📂 仓库结构

~~~text
AMR_BACKDOOR/
│
├── RQ1/                     # 正常 AMR 基线
├── RQ2/
│   ├── Seq/                 # 序列表征攻击
│   ├── Feat/                # 特征表征攻击
│   └── Img/                 # 星座图攻击
├── RQ3/                     # 后门检测与防御
├── RQ4/                     # 多模态 AMR
├── datasets/                # 星座图数据生成
├── rmlmodel/
│   ├── Sequence/
│   └── Image/
├── mltools.py
├── trigger_config.py
├── README.md                # 中文版 / GitHub 默认展示
└── README_EN.md             # 英文版
~~~

---

# 🚀 快速开始

## 1. 克隆仓库

~~~bash
git clone https://github.com/Xu0225/AMR_BACKDOOR.git
cd AMR_BACKDOOR
~~~

## 2. 创建环境

原项目主要基于 **Python 3.8** 开发。

~~~bash
conda create -n amr_backdoor python=3.8 -y
conda activate amr_backdoor
~~~

代码主要依赖：

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

> 原仓库未冻结完整依赖版本。复现实验时建议优先构建兼容 Python 3.8 的 TensorFlow / Keras 环境。

---

# 📡 数据集

实验主要基于 **RadioML 2016.10A**。

原始数据集项目：

https://github.com/radioML/dataset

下载后，需要根据本机路径修改：

~~~text
trigger_config.py
RQ1/main.py
~~~

## 生成星座图

~~~bash
cd datasets
python run_generation.py
~~~

---

# ⚙️ 复现实验说明

仓库保留了原始研究阶段代码，因此部分脚本仍包含开发机器上的绝对路径，例如：

~~~text
D:/zhaixu/Thesis_Code/
~~~

以及：

~~~text
/root/zx/Thesis_Code/
~~~

在新机器上运行前，建议优先检查：

~~~text
trigger_config.py
RQ1/main.py
RQ2/Seq/main.py
RQ2/Feat/main.py
RQ2/Img/img_bd_attack.py
~~~

后续如继续工程化，可统一为 **config.yaml / .env / CLI 参数**，提升项目的可移植性和复现性。

---

# 🧠 研究范围

~~~mermaid
mindmap
  root((AMR 安全))
    无线信号处理
      I/Q 信号
      SNR
      频谱
      星座图
    机器学习
      CNN
      RNN / GRU / LSTM
      树模型
      多模态学习
    AI 安全
      数据投毒
      后门攻击
      触发器设计
      后门检测
      模型修复
~~~

---

# 🔬 为什么要研究 AMR 后门？

AMR 模型可用于智能频谱监测、认知无线电、电磁环境感知和自动化通信系统。

一个被植入后门的模型可能：

- 在常规测试中表现完全正常；
- 在正常无线信号上保持较高识别率；
- 一旦出现特定触发模式，却稳定输出攻击者预设的错误类别。

因此，只看普通准确率并不足以判断模型是否安全。

本项目同时关注：

> **“模型在正常情况下是否仍保持正常性能？” —— CA**

以及：

> **“隐藏触发器出现后，后门行为是否会被激活？” —— ASR**

---

# ⚠️ 研究与使用说明

本仓库用于学术研究、AI 安全评测、AMR 鲁棒性研究、后门防御实验与可复现实验。请仅在获得授权的数据、模型和系统环境中开展相关测试。

---

<div align="center">

## ⭐ 如果这个项目对你的研究有所帮助

欢迎 Star，让更多做无线通信、AMR 与 AI 安全研究的人发现它。

<br/>

**探索无线智能与可信人工智能之间的边界。**

<br/>

[English README](./README_EN.md)

<br/><br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:8B5CF6,50:6366F1,100:06B6D4&height=120&section=footer"/>

</div>
