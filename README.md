# Task-Guided YOLOv8 for VEGA RISC-V

**DVCon India 2026 Design Contest - Team Dexters**

This repository contains the software/ML stack for a hardware-optimized, task-conditioned object detection system. Built specifically for deployment on the **VEGA RISC-V Processor** and **Genesys-2 FPGA**, this model dynamically filters visual features based on natural language tasks (e.g., "What should I use to cut a paper?").

Through rigorous hardware-software co-design, the model achieves state-of-the-art accuracy while compressing its memory footprint via **2:4 Structured Sparsity** and **Mixed-Precision Quantization (INT8/INT16)**.

---

## Setup & Installation

Due to strict C++ bindings in the quantization libraries, we use `uv` for lightning-fast and deterministic Python environment management.

### 1. Install `uv` (Python Package Manager)

* **For Linux and macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```


* **For Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

```



### 2. Install Dependencies

Create the environment and sync all required packages (PyTorch, Ultralytics, Brevitas, etc.):

```bash
uv venv --python 3.11
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
uv sync

```

### 3. Dataset Preparation

We train on the `COCO-Tasks` dataset. You must download the images and annotations separately.

1. Download **COCO 2014 Train & Val Images** from the official COCO website and place them in `coco/train2014/`.
2. Download the **COCO-Tasks Annotations** from this [Kaggle Mirror](https://www.kaggle.com/datasets/unnathchittimalla/coco-annotations) (bypassing GitHub LFS limits) and place the JSON files in `coco-tasks/annotations/`.

---

## Core Architecture & Logic

Our pipeline bridges the gap between massive AI models and strictly constrained edge hardware. We eliminate the need for an on-device LLM by using O(1) memory lookups and element-wise feature gating.

### 1. Task-Guided Fusion Engine

We intercept the feature maps of a YOLOv8n backbone. A custom Fusion Engine multiplies these visual features with pre-computed text embeddings.

**Backbone Channel Structure:**
*Note: YOLOv8n channels are 0.25x the typical YOLOv8 channels.*

* `Conv (0-P1/2)`: 64 channels (Downsampling)
* `Conv (1-P2/4)`: 128 channels (Downsampling)
* `C2f (2)`: 128 channels
* `Conv (3-P3/8)`: 256 channels (Downsampling) → **Fusion Point 1**
* `C2f (4)`: 256 channels
* `Conv (5-P4/16)`: 512 channels (Downsampling) → **Fusion Point 2**
* `C2f (6)`: 512 channels
* `Conv (7-P5/32)`: 1024 channels (Downsampling) → **Fusion Point 3**
* `C2f (8) / SPPF (9)`: 1024 channels

### 2. Hardware-Software Co-Design Implementations

* **O(1) BRAM Lookup:** Natural language tasks are pre-processed into static INT8 embeddings and stored in Block RAM (BRAM). The FPGA simply looks up the task vector at runtime.
* **Hardware-Aligned Precision:** Classification heads are quantized to **INT8** for maximum throughput, while bounding box coordinate heads are kept at **INT16** to prevent spatial misalignment.
* **BatchNorm Folding:** To save DSP slices and LUTs on the FPGA, all `BatchNorm2d` layers are mathematically folded directly into the adjacent `Conv2d` weights before quantization.

---

## Execution Pipeline

The two major modifications—Custom Fusion Layers and 2:4 Sparsity—must be trained together. We utilize a multi-stage pipeline to prevent the catastrophic "Model Shock" that occurs when severely pruning and quantizing a network simultaneously.

Run the pipeline sequentially using the following commands:

### Phase 1: Pre-compute Embeddings & Sanity Check

Start by extracting the task instructions and generating both FP32 and symmetric INT8 tensors.

```bash
uv run extract_task_embeddings.py
uv run view_embeddings.py   # Optional: View created tensor embeddings

```

You can verify the dense architecture and sparsity logic by running a dry test:

```bash
uv run model.py

```

*Expected Output: "Successfully applied 2:4 sparsity mask to 67 layers! ... TEST SUCCESS! Data flowed through the network without crashing."*

### Phase 2: Sparsity & Healing

Applies the 2:4 mask (forcing 50% of weights to strictly 0.0) and fine-tunes the surviving dense weights on the COCO-Tasks dataset.

```bash
uv run train.py
# Outputs: yolov8n_sparse_healed.pt

```

### Phase 3: Quantization-Aware Training (QAT)

Converts the healed model to Brevitas `QuantConv2d` layers and trains the network for a few epochs to survive simulated INT8/INT16 integer rounding operations.

```bash
uv run train_qat.py
# Outputs: yolov8n_sparse_qat.pt

```

### Phase 4: Hardware Extraction & Export

Runs the final calibration pass, mathematically folds the BatchNorm layers, compresses the data using the 2:4 protocol bitmasks, and generates the firmware payload.

```bash
uv run export_hardware.py
# Outputs: vega_yolo_weights.h

```

---

## Hardware Handoff

Once Phase 4 is complete, the software stack is finished.

**Instructions for the Firmware Team:** Copy `vega_yolo_weights.h` and `task_embeddings_int8.pt` directly into the VEGA RISC-V C-project firmware. The FPGA DSP logic must be programmed to decode the 4-bit `_masks` array to route the compressed `_weights` array accurately at runtime.