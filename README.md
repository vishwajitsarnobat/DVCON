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