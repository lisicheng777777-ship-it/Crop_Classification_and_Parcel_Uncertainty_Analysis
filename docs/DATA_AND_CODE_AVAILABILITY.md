# Data and code availability

The [code repository](https://github.com/lisicheng777777-ship-it/Crop_Classification_and_Parcel_Uncertainty_Analysis) and [data/model release v1.0.0](https://github.com/lisicheng777777-ship-it/Crop_Classification_and_Parcel_Uncertainty_Analysis/releases/tag/v1.0.0) are publicly accessible.

## Manuscript statement

The source code, processed experimental data, fixed data splits, archived predictions and reference metrics supporting this study are publicly available at https://github.com/lisicheng777777-ship-it/Crop_Classification_and_Parcel_Uncertainty_Analysis. Full-resolution input rasters, original vector data and 350 trained model checkpoints are available in release v1.0.0 at https://github.com/lisicheng777777-ship-it/Crop_Classification_and_Parcel_Uncertainty_Analysis/releases/tag/v1.0.0. File-level SHA-256 manifests and scripts for data restoration and numerical reproduction are included. The source code is licensed under the GNU General Public License version 3 (GPL-3.0).

## Downloads and verification

Download all nine crop-data.zip parts into release_assets/ and run:

```bash
python scripts/data_archive.py restore --directory release_assets
```

With the updated download manifest, automatic downloading is also available:

```bash
python scripts/data_archive.py restore --directory release_assets --download
```

On 2026-09-20, all nine public download endpoints were accessible and GitHub's reported asset sizes and SHA-256 digests matched the locally verified archive parts. This check did not re-download the complete 17.46 GB archive. The restoration script verifies downloaded parts and extracted files before use.

The v1.0.0 tag preserves its original source snapshot; later documentation and download-link updates are on main. Cite the exact source commit or a subsequent version tag when using those updates.

Data, weights and result-table reuse terms remain separate from the code license. No data license or DOI is asserted here. The feature-name discrepancy documented in METHODS_AND_LIMITATIONS.md must be resolved independently in the manuscript.
