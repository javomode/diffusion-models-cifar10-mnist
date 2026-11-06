# Evaluating Conditional Image Generation and Classifier-Free Guidance in Diffusion Models

DSC 180A checkpoint. We generate CIFAR-10 images with NVIDIA's pretrained EDM models (Karras et al., 2022), measure them with label accuracy and FID, and train small EDM networks on CIFAR-10 subsets with classifier-free guidance (CFG).

**Status:** CIFAR-10 parts 1-6 are done. The MNIST CFG network is in progress and has no results yet.

## Repository contents

| Path | What it is |
|---|---|
| `cifar10_edm_lab_jadenvo.ipynb` | The notebook with all code (parts 1-6) |
| `cifar10_edm_lab_jadenvo_out.ipynb` | The same notebook, executed, with outputs saved |
| `outputs/cfg_sweep_n1000.json` | Exact accuracy, FID and FID_class for each w (1,000 images per w) |
| `outputs/cfg_w*_n1000.npz` | The generated images and requested labels for each w |
| `run.log` | Log of the run, including training times and loss/FID per snapshot |
| `figure/` | The PNGs used in the report |
| `notes.md` | Answers to part 3's questions and other notes |

Not in the repo: `downloads/` (about 720 MB, re-created by the notebook), the training checkpoints `outputs/subset*_update*.pt` (about 27 MB each), and `outputs/cifar10-32x32-n50000.zip` (155 MB, only needed for EDM's `train.py`).

## Setup

1. Get a GPU. We ran on UCSD DSMLP. Colab also works: choose Runtime, then Change runtime type, then a GPU. On a non-NVIDIA machine the notebook sets `QUICK = True` and cuts every step down, which only tests the code.
2. Install the packages the notebook imports: `torch`, `numpy`, `matplotlib`, `pandas`, `pyarrow`, `pillow`. For EDM's own `train.py` (optional), also `click` and `psutil`.
3. Put `cifar10_edm_lab_jadenvo.ipynb` in an empty working folder and open it.

## Data downloads

The first code cells download everything and check each file against a SHA-256 checksum, so there is nothing to download by hand. They fetch about 720 MB into `downloads/`:

- EDM's pretrained CIFAR-10 networks, unconditional and conditional (`edm-cifar10-32x32-uncond-vp.pkl`, `edm-cifar10-32x32-cond-vp.pkl`)
- EDM's FID reference statistics (`cifar10-32x32.npz`) and Inception network (`inception-2015-12-05.pkl`)
- A CIFAR-10 ResNet-56 classifier from chenyaofo/pytorch-cifar-models, used only to score images
- CIFAR-10 itself, from Hugging Face's copy (`uoft-cs/cifar10`)
- EDM's code (NVlabs/edm, commit `008a4e5316c8e3bfe61a62f874bddba254295afb`)

MNIST will be downloaded with `torchvision.datasets.MNIST` once that part is run.

## Reproducing the results

Run the notebook from top to bottom on a GPU:

```
jupyter nbconvert --to notebook --execute cifar10_edm_lab_jadenvo.ipynb --output cifar10_edm_lab_jadenvo_out.ipynb
```

(or press Run All in Jupyter or Colab). Generated images are saved in `outputs/`, so a rerun reuses them. The long steps take about 54 minutes each for the two training runs (20,000 updates each, including FID at each snapshot).

The notebook displays its plots with `plt.show()` and does not save them. We saved each figure from the notebook output into `figure/` by hand, so the table below says which notebook cell produces each one.

## Figure and table guide

| Report item | File | Notebook part and cell |
|---|---|---|
| Fig. 1: unconditional samples | `figure/1uncond_model.png` | Part 1, the cell calling `edm_sample` with `uncond` (seed 0) |
| Fig. 2: unconditional trajectory | `figure/1uncond_sample_traj.png` | Part 1, `show_trajectory` on `uncond` |
| Fig. 3: conditional samples, one class per row | `figure/2cond_samples.png` | Part 2, the grid cell (seed 1) |
| Fig. 4: conditional trajectory, w = 1 | `figure/3cond_gen_w1.png` | Part 3, "Your turn" cell (class dog, seed 0) |
| Fig. 5: CFG trajectory, w = 4 | `figure/3cond_gen_w4.png` | Part 3, "Your turn" cell (class dog, seed 0) |
| Fig. 6: CFG sweep plots | `figure/5cfg_sweep_n1000.png` | Part 5, the sweep cell (2x2 plot) |
| Table 1: accuracy, FID, FID_class against w | `outputs/cfg_sweep_n1000.json` | Part 5, the same cell, which prints one line per w and writes the JSON |
| Fig. 7: CFG grid, own network, 100 images | `figure/6cfg_one_net_n100.png` | Part 6, "The effect of CFG with your network" |
| Fig. 8: CFG grid, own network, 1,000 images | `figure/6cfg_one_net_n1000.png` | Part 6, same cell |
| Fig. 9: training loss and FID across snapshots | `figure/6loss_fid_snapshots.png` | Part 6, "Loss and FID across snapshots" |
| Fig. 10-11: samples next to nearest training images | `figure/6snapshot_n100.png`, `figure/6snapshot_n1000.png` | Part 6, "Is it memorization?" |
| Fig. 12-13: samples at each snapshot | `figure/6uncond_n100.png`, `figure/6uncond_n1000.png` | Part 6, the cell after the loss plot |

## Results so far

CFG sweep with the two pretrained networks, 1,000 images per w, balanced requested classes (from `outputs/cfg_sweep_n1000.json`):

| w | Label accuracy | FID | FID_class |
|---|---|---|---|
| 0 | 11.5% | 31.19 | n/a |
| 1 | 95.5% | 31.33 | 112.12 |
| 1.5 | 98.7% | 31.44 | 107.62 |
| 2 | 98.8% | 33.05 | 108.89 |
| 3 | 99.5% | 36.03 | 112.45 |
| 4 | 98.9% | 39.03 | 117.53 |

FIDs from 1,000 samples are much higher than published values (EDM reports 1.97 and 1.79 with 50,000), so compare them only with each other.

Small-network training on 100 and 1,000 CIFAR-10 images (7.0M parameters, batch 64, Adam at 2e-4, label dropout 0.1, 20,000 updates): the loss falls throughout, but FID of 1,000 unconditional samples rises across snapshots (100 images: 74.4 to 111.6; 1,000 images: 94.1 to 129.9). The full per-snapshot numbers are in `run.log`.

## Still to do

- MNIST CFG network (MNIST guidance lab): label accuracy and FID for w = 0, 1, 2, 4.
- Optional: a full EDM training run on all 50,000 CIFAR-10 images with `train.py`.

## References

- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022. Code: https://github.com/NVlabs/edm
- Ho and Salimans, "Classifier-Free Diffusion Guidance", 2022.