# pulmonary embolism detection from ct scans
## authors
- [@sultanaZabin](https://github.com/sultanaZabin)
- [@RubaaAlghamdi](https://github.com/RubaaAlghamdi)
##

a deep learning pipeline that decides whether a patient has a **pulmonary embolism (pe)**, a blood clot in the lung arteries, from a ct pulmonary angiography scan. the priority is **high recall**: missing a sick patient is worse than a false alarm.


> ⚠️ research prototype for an academic project. **not a medical device** and not for clinical use.

## data
[rsna str pulmonary embolism detection](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection) (kaggle, 2020), ct scans from five international centers, labeled by expert radiologists.

- **3000 patients** used (indeterminate exams excluded), ~31% with pe
- split **by patient**: 70% train / 15% validation / 15% test, so no patient appears in two sets
- labels exist per slice (`pe_present_on_image`) and per patient
- the data is **not included** in this repo (competition rules). `picked.csv` lists the exact patients used

## approach
1. **preprocessing:** each scan is windowed to highlight blood and clots (center 100, width 700), resized to 256×256, and reduced to 64 evenly spaced slices
2. **slice model (2.5d):** a cnn scores each slice for "clot / no clot", using 3 neighboring slices as the 3 input channels for some 3d context
3. **augmentation:** small zoom, shift, ±5° rotation, contrast, brightness and noise (no flips, chest anatomy isn't symmetric)
4. **imbalance:** only ~5% of slices contain a clot, handled with a weighted loss
5. **patient decision:** patient score = average of their 5 most suspicious slices
6. **threshold:** chosen on validation to reach 90% recall, then applied unchanged to test

## models compared (validation, patient level)
| model | setup | val auc |
|---|---|---|
| pe_ct_cnn | small cnn trained from scratch (baseline) | see notebook |
| resnet18 | imagenet-pretrained, fine-tuned, early layers frozen | 0.705 |
| **efficientnet-b0** | imagenet-pretrained, fine-tuned, early layers frozen | **0.729** |

## final result (450 unseen test patients)
efficientnet-b0 + top-5 average, cutoff 0.68 (chosen on validation):

| metric | value |
|---|---|
| auc | 0.778 |
| recall | 0.90 (126 / 140 sick caught) |
| precision | 0.38 |
| specificity | 0.33 |
| f1 | 0.53 |

## repository structure
```
├── final committed project/      ← the actual project: everything needed to reproduce the results
│   ├── training_notebook.ipynb   ← full pipeline: preprocessing, training of all 3 models, evaluation, grad-cam, error analysis
│   ├── models/
│   │   ├── best_effnet.pt        ← final model (efficientnet-b0)
│   │   ├── best_resnet18_frozen.pt
│   │   └── best_pe_ct_cnn.pt     ← baseline
│   └── picked.csv                ← the 3000 patients used
├── extra code                    ← additional / experimental code, not part of the final pipeline
└── README.md
```

**start with `final committed project/`.** `extra code` holds extra work kept for reference; the results above don't depend on it.

## how to run
the notebook was built on **kaggle** (gpu t4): create a notebook from the competition page so the data is attached, then run the cells in order. preprocessing takes ~1 hour, each model ~40 min to train.

to load the final model:
```python
import torch, torchvision
model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
model.load_state_dict(torch.load("final committed project/models/best_effnet.pt", map_location="cpu"))
model.eval()
```

## demo app
a gradio app (separate notebook) lets you **upload a ct slice** or **point a camera** at one, and returns a clot probability plus a grad-cam heatmap of where the model looked. note: a single image gives a **per-slice** answer, not a per-patient diagnosis.

## limitations
- only 64 of ~250 slices per patient are kept, so some clots are lost before the model sees them
- at 90% recall the model still flags many healthy patients
- scores are not calibrated probabilities (class weighting inflates them)
- hyperparameters were set to common defaults; systematic tuning is future work
- validation was also used to pick the best epoch, so validation numbers are slightly optimistic; test gives the unbiased estimate

## future work
keep more slices per patient, a sequence model over slice features, ensembling the slice models, hyperparameter tuning, and comparing with a 3d ct foundation model (colipri).

## citation
Colak, E. et al. *The RSNA Pulmonary Embolism CT Dataset.* Radiology: Artificial Intelligence, 2021.
