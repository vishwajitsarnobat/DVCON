### First install uv, python package manager
- For Linux and Mac OS:
    curl -LsSf https://astral.sh/uv/install.sh | sh
- For Windows:
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

### Download all the needed dependencies:
- uv sync

### Main flow:
- Start with creating embeddings, by running extract_task_embeddings.py
- To view all created tensor embeddings, run view_embeddings.py

### Detailed Backbone Channel Structure (Typical)
- Conv (0-P1/2): 64 channels (Downsampling)
- Conv (1-P2/4): 128 channels (Downsampling)
- C2f (2): 128 channels
- Conv (3-P3/8): 256 channels (Downsampling)
- C2f (4): 256 channels
- Conv (5-P4/16): 512 channels (Downsampling)
- C2f (6): 512 channels
- Conv (7-P5/32): 1024 channels (Downsampling)
- C2f (8): 1024 channels
- SPPF (9): 1024 channels 

But, yolov8n is 0.25 times typical yolov8n.

### Testing model.py with pruning layers
vishwajit@fedora ~/Workspace/DVCON ±main⚡ » uv run model.py             2 ↵
Using device: cuda

Initializing TaskAwareYOLO...
Loading base YOLOv8n model...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yo
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt to 'yolov8n.pt': 100% ━━━━━━━━━━━━ 6.2MB 2.7MB/s 2.3s

Applying 2:4 structured sparsity
Successfully applied 2:4 sparsity mask to 67 layers!

Running forward pass...
TEST SUCCESS! Data flowed through the network without crashing.
Head returns tuple!
Final output shape: torch.Size([1, 84, 8400])
Meaning: (Batch Size, [Bounding box coordinates + class scores], Anchors)

- We applied 2:4 sparsity pruning, which makes 50% weights of the model to be 0.
- We need to train the model, currently 2 modifications have been done, first one being addition of custom layer for filtering out unwanted detections and second one being 2:4 sparsity. (training on COCO dataset).
- The 2 changes need to be trained together as individual training will be almost useless.

- The COCO-tasks annotations and COCO dataset needs to be downloaded.
