## PIGCN

This repository contains the source code and preprocessed dataset for the **WWW 2026 (<font color=red>acceptance rate: 20.1% (676 out of 3370), Oral paper</font>)** "[PIGCN: Physics-Inspired Graph Convolution Networks for Heterogeneous Social Event Detection](https://ieeexplore.ieee.org/)".

---
## Method Architecture

![PIGCN Offline Model Architecture](PIGCN_Framework.jpg)

---
## Run PIGCN

### Install Package Dependencies

    pip install requirements.txt

### Offline Social Event Detection

1) data_processing. to generate the initial textural features and indices for the messages.
```bash
cd data_process
python S2_generate_initial_features.py
python S3_save_edge_index.py
```

2) run offline detection.
```bash
python run_offline_model.py
```

### Incremental Social Event Detection

---

## PIGCN Visualization

### Wave-based Oscillatory Propagation

For an intuitive illustration of wave-based oscillatory propagation dynamics in social message diffusion, you may find it helpful when seeing the final figure in the WWW’26 paper: “Effective and Unsupervised Social Event Detection and Evolution via RAG and Structural Entropy.”

![Social Event Evolution Visualization](Social_Event_Evolution.png)

---

## Citation

Yongsheng Yu, Congbo Ma, Zitai Qiu, Shan Xue, Jian Yang, and JiaWu. 2026. PIGCN: Physics-Inspired Graph Convolution Networks for Heterogeneous Social Event Detection. In Proceedings of the ACM Web Conference 2026 (WWW ’26), April 13–17, 2026, Dubai, United Arab Emirates. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3774904.3792354 (<font color=red>oral paper</font>)
