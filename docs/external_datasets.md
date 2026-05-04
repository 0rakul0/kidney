# External renal ultrasound datasets

This note tracks candidate external datasets for expanding the renal ultrasound
pipeline. The main goal is to separate three different uses:

- kidney segmentation pretraining or pseudo-labeling;
- robustness to renal pathology such as stones;
- classification of parenchymal alteration or kidney failure as a proxy task
  related to fibrosis.

External datasets should not be mixed into the final experiments without a
record of source, license, modality, labels, and intended use.

## Priority candidates

### Kaggle

| Priority | Kaggle slug | Modality | Labels | Suggested use | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `gurjeetkaurmangat/kidney-ultrasound-images-stone-and-no-stone` | Renal ultrasound | Normal vs Stone | Pseudo-label kidneys, pretrain/fine-tune segmentation, robustness study | Large ultrasound set. Useful for segmentation diversity, not direct fibrosis labels. |
| 2 | `zaynebnouiri/renal-data` | Renal ultrasound | Kidney failure metadata/labels when available | Explore parenchymal/renal-failure classification | More relevant to the fibrosis direction, but labels and license must be inspected before use. |
| 3 | `siatsyx/ct2usforkidneyseg` | Synthetic ultrasound from CT | Kidney segmentation labels | Segmentation pretraining or ablation | Synthetic domain; use as auxiliary data only. |
| 4 | `ignaciorlando/ussimandsegm` | Abdominal ultrasound and simulated ultrasound | Organ segmentation labels including kidney | Segmentation pretraining and domain robustness | Includes real scans, but only a subset has manual masks. |

### Non-Kaggle sources

| Priority | Source | Modality | Labels | Suggested use | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | [Open Kidney Ultrasound Data Set / kidneyUS](https://github.com/rsingla92/kidneyUS) | 2D B-mode renal ultrasound | Kidney capsule, cortex, medulla, central echogenic complex, view, native/transplant | Main renal reference, multiclass parenchyma segmentation, feature extraction | This is the most aligned dataset. It requires registration for images and has a non-commercial license. |
| 2 | [TRUSTED](https://springernature.figshare.com/articles/dataset/TRUSTED_The_Paired_3D_Ultrasound_and_CT_Human_Data_for_Kidney_Segmentation_and_Registration_Research/27981050) | 3D transabdominal ultrasound paired with CT | Kidney segmentations and anatomical landmarks | Auxiliary segmentation/registration reference | Large 14.89 GB download and different 3D domain; useful, but not a direct drop-in for the current 2D pipeline. |
| 3 | [Abdominal Ultrasound Image Dataset for Organ Classification and Disease Detection](https://scholarsjunction.msstate.edu/research-data/5/) | Abdominal ultrasound | Organ classes, anomaly classes, patient-wise metadata in one folder | External kidney-image pool for pseudo-labeling and possible abnormal/normal proxy study | 5,005 images from 563 patients; includes kidney class and normal/abnormal folders; CC BY 4.0. |
| 4 | [Ultrasound Open Access Datasets directory](https://ultrasound-open-access.nidusai.ca/) | Ultrasound directory | Index of open datasets | Dataset discovery and license tracking | Curated NIDUS/RadOSS directory with 96 ultrasound datasets. |
| 5 | [CGPxy Ultrasound-Dataset](https://github.com/CGPxy/Ultrasound-Dataset) | Renal ultrasound | Release agreement and references; images not stored directly in git | Request-access source for 200 kidney ultrasound images | Cloned copy only contains README and release agreement; dataset requires sending the form by email. |
| 6 | [Clinical Ultrasound Image Repository](https://clinical-ultrasound-image-repository.s3.amazonaws.com/) | Mixed clinical ultrasound DICOM | Mixed repository | Large unlabeled pool if kidney/abdominal cases can be filtered | Needs careful filtering, de-identification/license review, and manifesting. |

### Hugging Face search result

The first Hugging Face search pass did not find a strong renal ultrasound
segmentation dataset. Results for `kidney` were mostly synthetic CKD tabular
data, CT, text, or unrelated image datasets. Keep Hugging Face as a monitoring
source, but it is not currently the strongest source for this thesis dataset.

## Lower-priority or modality-mismatch candidates

| Kaggle slug | Modality | Suggested status |
| --- | --- | --- |
| `safurahajiheidari/kidney-stone-images` | CT / mixed stone detection | Do not use for the main ultrasound thesis; possible object-detection reference only. |
| `murillobouzon/kssd2025-kidney-stone-segmentation-dataset` | CT | Not suitable for main ultrasound evidence. |
| `nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone` | CT | Useful only as background or separate modality experiment. |

## Recommended import policy

1. Download each dataset into `external_data/raw/<slug_name>/`.
2. Keep raw files untouched.
3. Create a manifest with file path, source slug, modality, class label, patient
   identifier if available, and license.
4. For ultrasound classification datasets without masks, generate pseudo-masks
   with the current best local segmenter.
5. Filter pseudo-masks before using them for training:
   - foreground area within plausible bounds;
   - one or two dominant connected components;
   - high model confidence;
   - visual audit of accepted and rejected examples.
6. Keep segmentation-expansion experiments separate from fibrosis/parenchyma
   experiments.

## Current priority order

1. Use the local `kidneyUS_images_25_june_2025/` data already present in this
   workspace as the primary parenchyma-oriented dataset.
2. Add the Kaggle stone/no-stone ultrasound dataset as a large external pool for
   pseudo-labeling and segmentation robustness.
3. Inspect `zaynebnouiri/renal-data` for kidney-failure or clinical proxy labels
   relevant to parenchymal disease.
4. Inspect the Mississippi State abdominal ultrasound dataset because it has a
   kidney class and abnormal/normal organization.
5. Request access to the CGPxy kidney ultrasound dataset if more manually curated
   renal ultrasound images are needed.
6. Treat TRUSTED as an optional auxiliary experiment because it is 3D and large.

## Local discovery artifacts

The NIDUS/RadOSS directory was downloaded locally to:

```text
external_data/indices/ultrasound-datasets.csv
```

A first renal/genitourinary/abdominal filter was saved to:

```text
external_data/indices/renal_ultrasound_candidates_from_nidus.csv
```

Downloaded source manifests are summarized in:

```text
docs/downloaded_external_data.md
```

## Fibrosis wording

Datasets labeled as stone, normal, or kidney failure do not provide direct
histological fibrosis labels. They can support:

- segmentation robustness;
- detection of renal abnormality;
- analysis of ultrasound markers associated with chronic parenchymal change.

They should not be described as proof of fibrosis detection unless linked to
biopsy, clinical report, specialist annotation, or a clearly defined proxy label.


