readme = """# USGS Streamflow Dataloaders

This repository provides two custom dataloaders for reading **USGS streamflow data**  
and preparing graph-based input tensors for training with the  
[Torch Spatiotemporal Library (TSL)](https://github.com/scdmlab/tsl).

---

## 🚀 Overview

These dataloaders are designed to integrate seamlessly with TSL-based models  
(e.g., HydroTGNN or other graph neural network architectures for hydrology).  
They handle the preprocessing, temporal slicing, and graph construction steps  
needed to train spatiotemporal models on USGS gauge datasets.

> 🔹 If you need access to the dataset or example splits, please contact  
> **📧 tsui5@wisc.edu**

---

## 📁 Repository Contents

| File | Description |
|------|--------------|
| `hydrology_dataset.py` | Custom dataset class for reading and preprocessing USGS streamflow data, generating temporal graph tensors compatible with TSL models. |
| `spin_hydrology.yaml` | General configuration file for training **any TSL model** on hydrology datasets. It defines data paths, graph connectivity, data splits, and training hyperparameters. |
| *(External)* [`tsl/`](https://github.com/scdmlab/tsl) | Full spatiotemporal modeling framework (forked for reproducibility and integration with these dataloaders). |

---

## 🔧 Usage Example

1. **Clone repositories**
   ```bash
   git clone https://github.com/your-username/USGS-Streamflow-Dataloaders.git
   git clone https://github.com/scdmlab/tsl.git
