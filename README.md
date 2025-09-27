# UPTor: Unified 3D Human Pose Dynamics and Trajectory Prediction for Human-Robot Interaction

[![arXiv](https://img.shields.io/badge/arXiv-2505.14866-b31b1b.svg)](https://arxiv.org/abs/2505.14866)  
We introduce a unified approach to forecast the dynamics of human keypoints along with the motion trajectory based on a short sequence of input poses. While many studies address either full-body pose prediction or motion trajectory prediction, only a few attempt to merge them. We propose a motion transformation technique to simultaneously predict full-body pose and trajectory key-points in a global coordinate frame. We utilize an off-the-shelf 3D human pose estimation module, a graph attention network to encode the skeleton structure, and a compact, non-autoregressive transformer suitable for real-time motion prediction for human-robot interaction and human-aware navigation. 

## Installation

**Clone the Repository**
```bash
git clone https://github.com/nisarganc/UPTor.git
cd UPTor
```
**Install Dependencies**
```bash
python -m venv create ./.uptor  
source ./.uptor/bin/activate  
pip install -r requirements.txt
```

## Dataset Setup
Download **Datasets** from [here](https://zenodo.org/records/16034038) and unzip it to the `root` of this repository with train and test split folders:  
```    
UPTor  
├── darko  
│   ├── test   
|   |  ├── 0001.npy  
|   |  └── ...  
│   ├── train 
|   |   ├── 0017.npy
|   |   └── ...
|   └── darko.yaml
├── cmu_mocap
|   └── ...
└── human_36m
    └── ...
``` 

## Dataset Visualization
```bash
python darko.py
python cmu_mocap.py
python human_36m.py
```

## Citation
If you use **UPTor** code in your research, please cite the following paper
```bibtex 
@misc{nilavadi2025uptorunified3dhuman,
  title         = {UPTor: Unified 3D Human Pose Dynamics and Trajectory Prediction for Human-Robot Interaction},
  author        = {Nisarga Nilavadi and Andrey Rudenko and Timm Linder},
  year          = {2025},
  eprint        = {2505.14866},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2505.14866},
}
```