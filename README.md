<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:06B6D4,45:6366F1,100:8B5CF6&height=220&section=header&text=AMR%20BACKDOOR&fontSize=52&fontColor=FFFFFF&fontAlignY=35&desc=Backdoor%20Attack%20%26%20Defense%20for%20Deep-Learning-based%20Automatic%20Modulation%20Recognition&descAlignY=55&descSize=16&animation=fadeIn"/>

# 📡 AMR BACKDOOR

### 面向深度学习自动调制识别的后门攻击与防御评测框架

<p>
一个面向研究的实验框架，用于系统研究 <b>自动调制识别（Automatic Modulation Recognition, AMR）</b> 模型在后门攻击下的安全性。
</p>

<p>
序列表征 · 统计特征 · 星座图 · 多模态融合
</p>

<p>
<a href="./README_EN.md">English</a> · <b>中文</b>
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

**干净信号 → 触发器注入 → 多视角表征 → AMR 模型 → CA / ASR → 后门防御**

</div>

---

## ⚡ 项目简介

基于深度学习的 **自动调制识别（AMR）** 已广泛用于智能无线通信系统，但其在 **训练阶段后门攻击（Backdoor Attack）** 下的安全性仍值得系统研究。

**AMR_BACKDOOR** 面向这一问题，构建了一套覆盖多种信号表征、AMR 模型、触发器设计与防御方法的实验框架。

项目围绕四个核心研究问题展开：

| 研究问题 | 关注内容 |
|---|---|
| **RQ1** | 典型深度 AMR 模型在干净数据上的基础性能如何？ |
| **RQ2** | 不同信号表征下，AMR 模型对后门攻击的脆弱性有何差异？ |
| **RQ3** | 现有后门检测与防御方法在 AMR 场景中效果如何？ |
| **RQ4** | 当 AMR 从单一表征扩展到多模态融合后，后门行为会如何变化？ |

---

## 🔥 项目亮点

<table>
<tr>
<td width="50%" valign="top">

### 📡 多视角信号表征

同一无线信号可从不同视角建模：

- **IQ**：原始同相 / 正交分量
- **AP**：幅度 / 相位表征
- **FFT**：频域表征
- **统计特征**
- **专家特征 / 高阶统计量**
- **星座图**
- **多模态融合**

</td>
<td width="50%" valign="top">

### 🧨 多种后门触发器

项目已实现多种触发器配置：

- BadNet
- Random Location
- Hanning
- Spectrum Shift
- Phase Shift
- Remapped AWGN
- Benign 基线

可在统一 AMR 实验流程下对不同触发器进行横向评测。

</td>
</tr>

<tr>
<td width="50%" valign="top">

### 🧠 模型库

**序列模型**

CNN2 · VTCNN2 · CLDNN · GRU · LSTM · CGDNN · MCLDNN · ResNet · DenseNet · ICAMC · MCNET · DAE ...

**特征模型**

CART · XGBoost · LightGBM

**图像模型**

CNN · VGG16 · ResNet50

</td>
<td width="50%" valign="top">

### 🛡️ 防御基准

项目包含多种典型防御方法：

- STRIP
- Spectral Signatures
- Activation Clustering
- Fine-Pruning
- t-SNE 可视化

可在同一 AMR 场景内完成攻击与防御对照实验。

</td>
</tr>
</table>

---

## 🧬 实验流程

~~~mermaid
flowchart LR
    A["📡 RadioML 2016.10A<br/>I/Q 信号"]

    A --> B1["〰️ 序列表征<br/>IQ · AP · FFT"]
    A --> B2["📊 特征表征<br/>Basic · Time · Expert"]
    A --> B3["🟣 星座图"]

    B1 --> C["🧨 后门触发器注入"]
    B2 --> C
    B3 --> C

    C --> D1["🧠 深度 AMR 模型"]
    C --> D2["🌳 特征机器学习模型"]
    C --> D3["🖼️ 图像模型"]

    D1 --> E["📈 评测"]
    D2 --> E
    D3 --> E

    E --> F1["干净准确率<br/>CA"]
    E --> F2["攻击成功率<br/>ASR"]
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

# 🧪 研究问题

## RQ1 — 干净条件下的 AMR 性能

RQ1 用于建立后门攻击实验之前的性能基线。

项目在 **RadioML 2016.10A** 上评测多类典型 AMR 深度模型，包括卷积、循环、混合与残差网络结构。

~~~bash
cd RQ1
python main.py
~~~

评测结果按不同 **信噪比（SNR）** 输出识别准确率。

项目中还保留了实验结果与模型文件：

~~~text
RQ1/
├── RQ1_results.xlsx
├── RQ1_results.pptx
├── results/
└── saved_model/
~~~

---

## RQ2 — AMR 后门攻击

RQ2 研究 **信号表征方式** 是否会改变 AMR 系统对后门攻击的敏感性。

### ① 序列表征

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

脚本会批量执行：

~~~text
表征 × 模型 × 触发器
~~~

并输出 **干净准确率（CA）** 与 **攻击成功率（ASR）**。

---

### ② 统计 / 专家特征表征

包含三类特征视图：

~~~text
basic
time
expert
~~~

支持模型：

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

特征提取与分析相关 Notebook：

~~~text
Feature_Extraction.ipynb
amr_feature_extraction.ipynb
CNN_FEAT.ipynb
CART_XGB_GBDT.ipynb
~~~

---

### ③ 星座图表征

项目还将 AMR 转化为图像识别任务，通过星座图进行建模。

支持模型：

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

这一分支用于研究后门攻击是否会从原始无线信号域迁移到 **视觉化信号表征**。

---

# 🛡️ RQ3 — 后门防御

项目集成了多种典型后门检测与缓解方法：

~~~text
RQ3/
├── AC/       # Activation Clustering
├── FP/       # Fine-Pruning
├── SS/       # Spectral Signatures
├── STRIP/    # STRIP
├── TSNE/     # 表征可视化
└── results/
~~~

Fine-Pruning 示例：

~~~bash
cd RQ3

python ./FP/fine_pruning.py     --TRIGGER_TYPE badnet     --POS_RATE 0.1     --EPOCH 100     --MODEL_NAME CNN2     --REP IQ
~~~

其他防御方法可直接进入对应目录执行。

> **说明**
>
> 当前 `run_defense.py` 是历史实验驱动脚本，其中多个 `command` 会被连续覆盖，因此实际执行的是最后一次赋值后的命令。为了保证实验可复现，建议分别运行各防御模块。

---

# 🔀 RQ4 — 多模态 AMR

RQ4 研究当 AMR 不再依赖单一表征，而是融合多种信号视角时，后门行为是否会发生变化。

仓库中包含：

~~~text
单模型推理
集成推理
基于 Stacking 的多模态融合
~~~

相关文件：

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

当前实验驱动默认采用 **Stacking 多模态融合流程**。

---

# 🧨 触发器配置

核心触发器代码位于：

~~~text
trigger_config.py
~~~

已实现：

| 触发器 | 核心思路 |
|---|---|
| **BadNet** | 在指定局部位置注入固定触发模式 |
| **Random Location** | 在不同时间位置随机注入触发模式 |
| **Hanning** | 对信号施加 Hanning 窗变换 |
| **Spectrum Shift** | 在频域中改变信号特征 |
| **Phase Shift** | 引入相位域偏移 |
| **Remapped AWGN** | 使用重映射噪声构造触发器 |
| **Benign** | 不注入恶意触发器，作为干净基线 |

实验脚本中默认投毒比例通常为：

~~~text
10%
~~~

---

# 📊 评测指标

项目主要使用两个核心指标。

### Clean Accuracy — CA

后门模型在正常测试样本上的识别准确率。

> CA 越高，说明模型在正常输入下越接近“看起来完全正常”。

### Attack Success Rate — ASR

当触发器出现时，模型将样本预测为攻击者目标类别的成功率。

> ASR 越高，说明后门被触发后的攻击效果越强。

同时，框架还会记录不同 **SNR** 条件下的模型表现，用于研究：

~~~text
信道质量 × 信号表征 × 模型 × 触发器
~~~

之间的交互关系。

---

# 📂 项目结构

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
│   ├── Seq/                 # 基于序列的 AMR 后门攻击
│   ├── Feat/                # 基于统计/专家特征的攻击
│   └── Img/                 # 基于星座图的攻击
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
├── README.md
└── README_EN.md
~~~

---

# 🚀 快速开始

## 1. 克隆项目

~~~bash
git clone https://github.com/Xu0225/AMR_BACKDOOR.git
cd AMR_BACKDOOR
~~~

## 2. 创建环境

原始项目主要基于 **Python 3.8** 开发。

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

> 原仓库未固定完整依赖版本。若要复现实验，建议优先构建与 Python 3.8 兼容的 TensorFlow / Keras 环境。

---

# 📡 数据集

实验主要基于：

### RadioML 2016.10A

原始数据集项目：

https://github.com/radioML/dataset

下载后需要配置以下脚本中的本地路径：

~~~text
trigger_config.py
RQ1/main.py
~~~

---

## 生成星座图数据

~~~bash
cd datasets
python run_generation.py
~~~

生成的星座图用于图像表征 AMR 实验。

---

# ⚙️ 可复现性说明

仓库保留了原始实验代码，因此部分脚本仍包含历史机器上的绝对路径，例如：

~~~text
D:/zhaixu/Thesis_Code/
~~~

以及：

~~~text
/root/zx/Thesis_Code/
~~~

在新机器复现实验前，建议全局搜索并替换为本机项目和数据集路径。

重点检查：

~~~text
trigger_config.py
RQ1/main.py
RQ2/Seq/main.py
RQ2/Feat/main.py
RQ2/Img/img_bd_attack.py
~~~

后续可进一步将这些绝对路径统一迁移到：

~~~text
config.yaml
.env
命令行参数
~~~

以提升项目可移植性。

---

# 🧠 研究范围

~~~mermaid
mindmap
  root((AMR Security))
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

# 🔬 为什么 AMR 后门值得研究？

AMR 模型可用于智能频谱监测、认知无线电、电磁环境感知与自主通信系统。

一个被植入后门的 AMR 模型可能：

- 在常规测试中表现正常；
- 在正常信号上保持较高识别准确率；
- 但一旦检测到隐藏触发器，就系统性输出攻击者指定结果。

因此，仅靠普通准确率很难发现后门。

这个项目同时关注两个问题：

> **“模型平时看起来正常吗？” —— CA**

以及：

> **“隐藏行为被触发后是否成功？” —— ASR**

---

# ⚠️ 研究与使用说明

本项目主要用于：

- 学术研究
- AI 安全评测
- AMR 鲁棒性研究
- 后门防御研究
- 可复现实验

请仅在你拥有授权的数据、模型和系统上开展相关实验。

---

<div align="center">

## ⭐ 如果这个项目对你的研究有帮助

欢迎点一个 Star，让更多人看到这个项目。

<br/>

**探索无线智能与可信 AI 的交叉边界。**

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:8B5CF6,50:6366F1,100:06B6D4&height=120&section=footer"/>

</div>
