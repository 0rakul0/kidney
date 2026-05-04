# Downloaded external data

This file documents the external data fetched or inspected for the renal
ultrasound project. Raw data is stored under `external_data/`, which is ignored by
Git because it contains large datasets and third-party files.

## Downloaded locally

### MONAI / NVIDIA Clinical Ultrasound Image Repository

- Source: <https://clinical-ultrasound-image-repository.s3.amazonaws.com/index.html>
- Local root: `external_data/raw/MONAI_ClinicalUltrasoundRepository/`
- License: CC-BY-NC 4.0
- Modality: clinical ultrasound DICOM
- Relevant subset: abdominal studies with renal, kidney, or retroperitoneal
  terms in metadata/study reference.
- Full repository size: not downloaded.
- Metadata downloaded:
  - `external_data/raw/MONAI_ClinicalUltrasoundRepository/meta-only/all-meta.json`
  - `external_data/raw/MONAI_ClinicalUltrasoundRepository/meta-only/all-meta.csv`
- Renal candidate index:
  - `external_data/indices/monai_clinical_ultrasound_renal_studies.csv`
  - 238 renal/retroperitoneal candidate studies found.
- Archive size estimate:
  - `external_data/indices/monai_renal_study_download_sizes.csv`
  - 238 candidate study zips.
  - Estimated total: 68.82 GB.
- Downloaded subsets:
  - The 238 renal/retroperitoneal candidate study archives were downloaded in
    multiple size-capped batches.
  - Already processed studies were skipped in later batches.
  - Total temporary zip download across all batches: approximately 68.82 GB.
  - Download manifest: `external_data/indices/monai_downloaded_renal_subset_manifest.csv`
  - The raw zip archives were removed after each lightweight PNG curation pass
    to reduce local storage use.

Suggested use:

- Extract DICOM frames from the downloaded studies.
- Filter non-kidney frames where possible.
- Generate pseudo-masks with the current renal segmenter.
- Use only accepted masks for segmentation robustness experiments.

Limitations:

- No manual kidney masks are provided.
- The repository is mixed clinical ultrasound; even renal study archives may
  contain frames that are not ideal kidney B-mode images.
- License is non-commercial.

Lightweight curation:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\curate_monai_renal_dicoms.py --clear-output
```

This converts eligible DICOM frames to PNG, copies per-study JSON/CSV metadata,
and writes:

- `external_data/processed/monai_renal_png/images/`
- `external_data/processed/monai_renal_png/metadata/`
- `external_data/processed/monai_renal_png/manifest.csv`
- `external_data/processed/monai_renal_png/summary.json`

By default, RGB/color DICOMs are skipped to avoid Doppler/color frames. Cine
DICOMs are reduced to at most three representative frames.

Current curated local subset after all MONAI renal/retroperitoneal candidates:

- accepted PNG images: 4,487;
- rejected DICOM/frame rows: 5,716;
- rejection reason: RGB/color skipped;
- processed PNG size: 702,033,551 bytes, approximately 669.51 MB;
- total processed folder size: approximately 676 MB.

### NIDUS/RadOSS Ultrasound Open Access Datasets directory

- Source: <https://ultrasound-open-access.nidusai.ca/>
- Local index:
  - `external_data/indices/ultrasound-datasets.csv`
- Filtered renal/genitourinary/abdominal candidates:
  - `external_data/indices/renal_ultrasound_candidates_from_nidus.csv`

Suggested use:

- Dataset discovery and license tracking.
- Not image data by itself.

### CGPxy Ultrasound-Dataset repository

- Source: <https://github.com/CGPxy/Ultrasound-Dataset>
- Local clone:
  - `external_data/raw/CGPxy__Ultrasound-Dataset/`
- Status:
  - Repository cloned successfully.
  - The kidney folder contains README/release agreement material, not the image
    files themselves.
  - Access to the 200 kidney ultrasound images requires completing the release
    agreement and sending it to the dataset maintainers.

Suggested use:

- Request access if more curated renal ultrasound images are needed.

## Not downloaded yet

### Kaggle datasets

Kaggle CLI is installed inside `.venv`, but Kaggle credentials are not present.
Place `kaggle.json` in `C:\Users\jeffe\.kaggle\kaggle.json` before downloading.

Configured Kaggle sources:

- `gurjeetkaurmangat/kidney-ultrasound-images-stone-and-no-stone`
- `zaynebnouiri/renal-data`
- `siatsyx/ct2usforkidneyseg`
- `ignaciorlando/ussimandsegm`

Download command:

```powershell
.\.venv\Scripts\python.exe engenharia_dataset\download_kaggle_datasets.py
```

### Mississippi State abdominal ultrasound dataset

- Source: <https://scholarsjunction.msstate.edu/research-data/5/>
- DOI: <https://doi.org/10.54718/LZXF6315>
- License: CC BY 4.0
- Contents described by source:
  - 5,005 abdominal ultrasound images.
  - 563 patients.
  - organ classification folders including kidney.
  - normal/abnormal folders.
  - patient-wise folder with diagnosis metadata.
- Status:
  - The page is reachable.
  - Automated download from the native content URL returned HTTP 403.
  - Manual browser download may be required.

### TRUSTED

- Source: <https://springernature.figshare.com/articles/dataset/TRUSTED_The_Paired_3D_Ultrasound_and_CT_Human_Data_for_Kidney_Segmentation_and_Registration_Research/27981050>
- License: CC BY
- Size listed by source: 14.89 GB.
- Status:
  - Not downloaded in this pass because it is a large 3D ultrasound/CT dataset
    and is auxiliary to the current 2D B-mode workflow.

## Provenance rules for thesis experiments

For any experiment that uses external data, record:

- source name and URL;
- license;
- download date;
- local path;
- original labels;
- whether labels are manual, clinical, heuristic, or pseudo-labels;
- whether the data is used for segmentation, pretraining, classification, or
  qualitative validation.

External pseudo-labeled data should be reported separately from manually labeled
data in all thesis tables.


