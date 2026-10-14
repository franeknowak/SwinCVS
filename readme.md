# SwinCVS: A Unified Approach to Classifying Critical View of Safety Structures in Laparoscopic Cholecystectomy

**Authors**:  
Franciszek Nowak, Evangelos B. Mazomenos, Brian Davidson, Matthew J. Clarkson  

--- 

## Overview

Welcome. This repository provides code necessary for reproduction of the SwinCVS publication. The work proposes a SwinV2+LSTM based architecture called SwinCVS, to classify three Critical View of Safety (CVS) criteria from an open access Endoscapes2023 dataset.  

## Implemented models

- **SwinV2 Backbone**: Pure SwinV2 backbone. Can be run on random weights or initialised using provided ImageNet weights.
- **SwinCSV (E2E, with multiclassifier)**: SwinCVS with end-to-end training and multiclassifier. Backbone weights initialised on ImageNet.
- **SwinCSV (E2E, without multiclassifier)**: SwinCVS with end-to-end training, but without multiclassifier. Backbone weights initialised on ImageNet.
- **SwinCSV (Frozen, without multiclassifier)**: SwinCVS where image encoding backbone is frozen. Suggested backbone weights pretrained on Endoscapes.

## Installation

- Clone this repository
- Confirm you have cuda enabled. In console type nvidia-smi. Our driver API details are:
NVIDIA-SMI 550.120 | Driver Version: 550.120 | CUDA Version: 12.4
- Install runtime API cuda 12.1 - Remember to add to path!
- Install dependencies:<br>
conda create --name swincvs python=3.9.19<br>
conda activate swincvs<br>
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia<br>
pip install -r requirements.txt<br>

## Usage

Script is run by executing SwinCVS.py from the root of the repository. Specific model training parameters are set within config/SwinCVS_config.yaml. Settings that specify model selection are:
- MODEL.LSTM: False - just SwinV2 backbone training, True - SwinCVS = SwinV2 with LSTM
- MODEL.E2E: False - backbone weights frozen, True - End-to-end training
- MODEL.MULTICLASSIFIER: False - does not add an additional classifier after backbone, True - adds a classifier after backbone, before LSTM 
- MODEL.INFERENCE: False - allows for training, True - skips all training, performs only testing on provided weights
- BACKBONE.PRETRAINED: 'str' - which backbone weights to load, imagenet or endoscapes<br>

The script automatically downloads the dataset and the weights. If you already have dataset downloaded please specify directory that contains 'endoscapes' folder with all the dataset data.<br>

After changing the config, execute the script by running the SwinCVS.py and specifying which config file to use (default below):<br>
python3 SwinCVS.py --config_path config/SwinCVS_config.yaml

**Note on metrics:** `avg_bal_acc` in the results json is the mean of the per-criterion balanced accuracies (`C1_bacc`, `C2_bacc`, `C3_bacc`). In earlier versions of this code it was the mean recall (sensitivity) over C1-C3; that value is still saved as `avg_recall`, so older results can be compared. Per-criterion values and mAP are unchanged.

## Testing

`python3 test.py` checks the metrics against scikit-learn, loads every config, builds each model variant, and runs `inference.py` and `train_safe.py` end-to-end on a small synthetic dataset (random weights, CPU/MPS/CUDA; no dataset or GPU needed, takes a few minutes). Run a single group with e.g. `python3 -m unittest test.TestMetrics`.

## Extensions: standalone inference and fine-tuning on new data

Contributed by [@supremekhadka](https://github.com/supremekhadka). These tools sit alongside the original pipeline above, which is unchanged. For full documentation, benchmark results and the original development history, see their fork: [supremekhadka/SwinCVS](https://github.com/supremekhadka/SwinCVS).

- **Weights**: `python3 download_weights.py` downloads all released weights up front, including the frozen SwinCVS weights (`SwinCVS_frozen_ENDP_sd5_bestMAP.pt`, hosted on [Hugging Face](https://huggingface.co/supremekhadka/SwinCVS)). A frozen-model training config is also provided: `config/SwinCVS_frozen_config.yaml`.
- **Inference** (`inference.py`): runs the released (or your own) SwinCVS weights on new data without the training pipeline. Runs on CUDA, Apple silicon (MPS) or CPU, picked automatically. Also writes per-frame/per-video timing and GPU memory/power usage.
```
python3 inference.py --dataset {endoscapes,safe} --config_path config/infer.yaml|config/infer_safe.yaml \
    --mode {e2e,frozen} --output_dir results/my_run [--csv_path ...] [--images_path ... | --video_root ...]
```
  - `--dataset endoscapes`: a csv with `vid, frame, C1, C2, C3` columns in consecutive blocks of 5 rows (label on the 5th row), images in one folder as `<vid>_<frame>.jpg`. Writes `result.csv` with confidences and `metrics.json`.
  - `--dataset safe`: a csv with `vid, frame, is_ds_keyframe` columns; each keyframe is predicted from itself and the 4 preceding frames. Images in `VIDEO_ROOT/<vid>/<vid>_<frame>.jpg`. Writes `result.csv` with confidences.
- **Fine-tuning** (`train_safe.py`): fine-tunes the released e2e or frozen SwinCVS on a SAFE-format dataset (`labels/1fps/splits/{train,val,test}.csv` with `C1, C2, C3` labels). Supports training only the last N backbone layers (`--train_last_layers N`) and optional wandb logging (`--no_wandb` to disable). Checkpoints can be passed to `inference.py --weights`.
```
python3 train_safe.py --config config/SwinCVS_safe_finetune_frozen.yaml --data_root /path/to/SAFE
python3 train_safe.py --config config/SwinCVS_safe_finetune_e2e.yaml --data_root /path/to/SAFE
```

## Citation

If you use this work in your research, please cite our paper:
Nowak, F., Mazomenos, E., Davidson, B., Clarkson, M., SwinCVS: A Unified Approach to Classifying Critical View of Safety Structures in Laparoscopic Cholecystectomy. Int J CARS (2025). https://doi.org/10.1007/s11548-025-03354-9
