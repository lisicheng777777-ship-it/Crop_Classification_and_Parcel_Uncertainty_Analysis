# Third-party notices and code license

The combined research software is distributed under GNU GPL version 3 (GPL-3.0-only); see [LICENSE](../LICENSE). Existing third-party copyright and attribution notices remain applicable. Data, trained weights, and result tables are not granted a license by this code-license declaration; their redistribution and reuse terms must be stated separately.

## TempCNN

The retained implementation derives from [BreizhCrops](https://github.com/dl4sits/BreizhCrops), whose LICENSE is GNU GPL version 3. Verified upstream revision: `6de796ed36a457c8520322d6110b8f2862fd8c25`.

- [Upstream implementation](https://github.com/dl4sits/BreizhCrops/blob/6de796ed36a457c8520322d6110b8f2862fd8c25/breizhcrops/models/TempCNN.py)
- [Upstream license](https://github.com/dl4sits/BreizhCrops/blob/6de796ed36a457c8520322d6110b8f2862fd8c25/LICENSE)
- Intermediate source: [DongbeiCrops](https://github.com/GenghisYoung233/DongbeiCrops); its README attributes its models to BreizhCrops. The local experimental copy matched the local DongbeiCrops TempCNN file after newline normalization. The historical intermediate commit is unknown; the revision above is the comparison reference, not a claimed original download date.

Comparison on 2026-09-17 established that the experimental implementation differs from the pinned BreizhCrops file only in comments and constructor defaults: kernel_size 7 to 3, hidden_dims 128 to 32, and dropout 0.18203942949809093 to 0.5. These pre-existing experimental modifications are preserved. On 2026-09-17 a license/provenance header was added; executable code was not changed.

BreizhCrops credits the original TempCNN work by Pelletier et al. (2019), Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series, Remote Sensing 11(5), 523, https://doi.org/10.3390/rs11050523, and https://github.com/charlotte-pel/temporalCNN. Preserve this scientific attribution alongside the software license. No claim of sole authorship is made for the inherited implementation.

## Other components

The data utility retains its existing attribution to Development Seed satellite machine-learning training material. Existing source notices are preserved. External Python packages are installed as dependencies and retain their own licenses; the project GPL declaration does not replace their licenses.

## Distribution

Distribute the complete corresponding software source, this license, and these notices together. Retain notices and identify subsequent modifications. No warranty is provided; see LICENSE. This records the verified TempCNN source and the selected code-distribution terms, not a certification of ownership of every dataset or dependency.
