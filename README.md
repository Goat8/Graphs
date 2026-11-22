# Graph Neural Networks for Spatio-Temporal Traffic Forecasting

This repository contains the code and results for Programming Assignment 3.

- Questions 1 and 2 use small NumPy scripts that run on any machine.
- Questions 3 to 6 use the San Diego (SD) subset of the LargeST traffic dataset.
- Questions 3 to 6 were run on Google Colab with a Tesla T4 GPU.

## Repository layout

| Path                                                | Question              | What it does                                                                                                                                      |
| --------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/Question1(b).py`                               | Q1(b)                 | Checks the base K+1 message encoding for every neighbourhood with K = 8.                                                                          |
| `src/Question2.py`                                  | Q2                    | Fits the symmetric layer and plots the two heatmaps, and checks parts (b) and (c).                                                                |
| `src/Question03.ipynb`                              | Q3                    | Downloads LargeST, extracts the 716 San Diego sensors, resamples to 15 minutes, and analyses the road graph.                                      |
| `src/Question_4_5_6.ipynb`                          | Q4, Q5, Q6            | Main experiment. Trains the temporal-only, fixed, adaptive and hybrid models with the specified layer H = ReLU(ÂZW_g), then runs the Q6 analysis. |
| `src/SKIP_CONNECTION_EXPERIMENT_Models_4_5_6.ipynb` | Additional experiment | Same notebook with a residual connection, H = ReLU(ÂZW_g) + Z.                                                                                    |
| `data/sd_meta.csv`                                  | Q3 to Q6              | Metadata for the 716 San Diego sensors, produced by the Q3 notebook.                                                                              |
| `results/Question03/`                               | Q3                    | Log, one-week flow plot, adjacency heatmap, and neighbour distribution.                                                                           |
| `results/Question4_5/`                              | Q4, Q5                | Training log, results table, and per-epoch training histories.                                                                                    |
| `results/Question06/`                               | Q6                    | Analysis log, prediction plots, validation curves, and learned-adjacency heatmaps.                                                                |

The results of the residual experiment are shown in the cell outputs of
`src/SKIP_CONNECTION_EXPERIMENT_Models_4_5_6.ipynb`.

## Requirements

- Python 3.10 or later
- `numpy`, `pandas`, `matplotlib`, `h5py`, `tables`, `torch`, `kagglehub`

Google Colab already provides all of these except `kagglehub`, which the Q3 notebook installs.

## How to reproduce

Run the steps in this order.

### Step 1: Questions 1 and 2 (local)

```bash
python "src/Question1(b).py"
python src/Question2.py
```

The first script prints a confirmation that all 164 neighbourhoods are decoded correctly.
The second script prints the fitted coefficients and saves `q2_heatmaps.png`.

### Step 2: Question 3 (Colab)

1. Open `src/Question03.ipynb` in Google Colab.
2. Run all cells.
   The first cell downloads the LargeST dataset from Kaggle with `kagglehub`.
   Kaggle may ask you to sign in.
3. The notebook writes these files to the `outputs/` folder:
   - `sd_dataset.npz`: the cleaned 15-minute flow, the missing-value mask, the timestamps, and the 716 × 716 adjacency
   - `sd_meta.csv`: the sensor metadata
   - `q3_results.txt` and the three Q3 figures

`LOW_MEMORY = True` reads the large California files in blocks.
This lets the notebook run within Colab's memory limit.

### Step 3: Make the dataset available to Questions 4 to 6

The training notebooks read the dataset from Google Drive at:

```
/content/drive/MyDrive/PA3/sd_dataset.npz
```

Use one of these options:

- Upload your own `outputs/sd_dataset.npz` from Step 2 to a Drive folder named `PA3`.
- Or use the prepared copy in the shared folder:
  https://drive.google.com/drive/folders/1ACHwvzYFSsmwTVkHzL23V9MXdFsGgZJf?usp=sharing
  Add a shortcut to it in "My Drive" and name the shortcut `PA3`.

### Step 4: Questions 4, 5 and 6 (Colab, GPU)

1. Open `src/Question_4_5_6.ipynb` in Google Colab.
2. Select a GPU runtime: Runtime → Change runtime type → T4 GPU.
3. Run the first cell. It mounts Google Drive and copies `sd_dataset.npz` into `outputs/`.
4. Upload `data/sd_meta.csv` from this repository into the Colab `outputs/` folder.
   The Question 6 cell needs it.
5. Run the remaining cells in order.

The training cell trains four models in sequence:

| Output block               | Question   | Model                                             |
| -------------------------- | ---------- | ------------------------------------------------- |
| `Training model: temporal` | 4(c)       | Temporal-only baseline, with no graph layer       |
| `Training model: fixed`    | 4(b), 5(a) | Road-distance graph, row-normalised as D⁻¹(A + I) |
| `Training model: adaptive` | 5(b)       | Learned graph, Softmax(ReLU(E₁E₂ᵀ))               |
| `Training model: hybrid`   | 5(c)       | 0.5 × fixed + 0.5 × adaptive                      |

All outputs are written to `outputs/`.
The whole notebook takes about 15 minutes on a T4 GPU.

### Step 5: Residual-connection experiment (optional)

1. Open `src/SKIP_CONNECTION_EXPERIMENT_Models_4_5_6.ipynb` in a fresh Colab runtime.
2. Repeat steps 2 to 4 of Step 4.
   Also upload `data/sd_meta.csv` into an `outputs_skip/` folder.
3. Run the remaining cells.

This notebook differs from the main one only in its settings cell:

```python
OUT_DIR = "outputs_skip"
SKIP_CONNECTION = True
```

Its outputs are written to `outputs_skip/`.

## Training configuration

All models share these settings.

| Setting          | Value                                                                      |
| ---------------- | -------------------------------------------------------------------------- |
| Random seed      | 42                                                                         |
| Input history    | 8 steps (2 hours)                                                          |
| Forecast horizon | 4 steps (1 hour)                                                           |
| Hidden dimension | 32                                                                         |
| Batch size       | 64                                                                         |
| Epochs           | 10                                                                         |
| Optimiser        | Adam, learning rate 0.001                                                  |
| Loss             | Masked MAE on normalised flow                                              |
| Normalisation    | Per-sensor z-score, using 2020 statistics only                             |
| Model selection  | Weights with the lowest validation MAE                                     |
| Splits           | Train 2020 / validation January–March 2021 / test April–December 2021      |
| Resampling       | Sum of three 5-minute counts (vehicles per 15 minutes)                     |
| Missing values   | Forward fill, then backward fill; excluded from loss and metrics by a mask |

Settings that differ between models:

| Model          | Adjacency                             | Node embedding size | Parameters |
| -------------- | ------------------------------------- | ------------------- | ---------- |
| Temporal-only  | None                                  | —                   | 11,588     |
| STGNN fixed    | Row-normalised road graph, D⁻¹(A + I) | —                   | 12,612     |
| STGNN adaptive | Softmax(ReLU(E₁E₂ᵀ)), row-wise        | 10                  | 26,932     |
| STGNN hybrid   | 0.5 × fixed + 0.5 × adaptive          | 10                  | 26,932     |

No top-k sparsification was used.
The hybrid weight α was fixed at 0.5.

## Hardware

| Item      | Value              |
| --------- | ------------------ |
| Platform  | Google Colab       |
| GPU       | NVIDIA Tesla T4    |
| CPU cores | 2                  |
| PyTorch   | 2.14.1 (CUDA 13.0) |
| NumPy     | 2.4.6              |

## Results

Test period: April–December 2021. Errors are in vehicles per 15 minutes.
Training time covers 10 epochs and excludes validation.

**Main experiment, specified layer H = ReLU(ÂZW_g):**

| Model         | Adjacency | Test MAE | Test RMSE | Training time (s) |
| ------------- | --------- | -------- | --------- | ----------------- |
| Temporal-only | None      | 62.64    | 98.92     | 195.1             |
| STGNN         | Fixed     | 101.53   | 155.13    | 208.6             |
| STGNN         | Adaptive  | 82.43    | 129.85    | 208.8             |
| STGNN         | Hybrid    | 79.73    | 124.45    | 209.0             |

**Additional experiment, residual layer H = ReLU(ÂZW_g) + Z:**

| Model         | Adjacency | Test MAE | Test RMSE | Training time (s) |
| ------------- | --------- | -------- | --------- | ----------------- |
| Temporal-only | None      | 62.71    | 99.01     | 194.8             |
| STGNN         | Fixed     | 59.54    | 93.66     | 209.7             |
| STGNN         | Adaptive  | 54.85    | 86.49     | 213.7             |
| STGNN         | Hybrid    | 56.44    | 88.97     | 216.1             |

## Reproducibility note

The random seed is fixed for every model.
Non-deterministic GPU operations still cause small differences between runs.
Retraining the temporal-only model with the same seed changed its test MAE from 62.64 to 62.71, a difference of 0.12%.
This is far smaller than the differences between models.

## AI usage

Generative AI (Chatgpt) was used as a learning and coding partner.
