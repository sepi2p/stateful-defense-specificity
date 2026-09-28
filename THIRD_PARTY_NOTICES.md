# Third-party material

## Included

`surro_models/cifar10_models/resnet.py` is taken unchanged from
https://github.com/kuangliu/pytorch-cifar (`models/resnet.py`), MIT licence:

    MIT License. Copyright (c) 2017 liukuang

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions: The above copyright
    notice and this permission notice shall be included in all copies or substantial portions of
    the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

## Not included (obtain from the sources)

| What | Source | Licence | Commit used |
|---|---|---|---|
| GWAD and GWAD+ detectors, Delta-Net weights | https://github.com/jpark04-qub/GWAD | GPL-3.0 | 822b258eb28c4c51eefa8d2cc87072991f750b74 |
| Blacklight (reference for the port) | https://github.com/Huiying-Li/blacklight | MPL-2.0 | b130337fe3607cb371a42cf2da74459659643f24 |
| VGG19-BN for CIFAR-10, definition and weights | https://github.com/SCLBD/BlackboxBench | CC BY-NC 4.0 | see `utils/load_models.py` |
| Robust ResNet-50 for CIFAR-10 (Engstrom2019Robustness) | RobustBench, https://github.com/RobustBench/robustbench | see source | downloaded by the `robustbench` package |
| ResNet-50 for ImageNet | torchvision (`ResNet50_Weights.IMAGENET1K_V1`) | see source | |
| `lime` 0.2.0.1, `captum` 0.9.0 | PyPI | BSD | |
| CIFAR-10, GTSRB, ImageNet (validation set) | the datasets' providers | see sources | |

`third_party/README.md` gives the paths at which the scripts expect the cloned repositories.
The detector of Lee, Fang and Chang (arXiv:2606.21592) has no released code;
`experiments/gate_trajectory_signatures/lfc_detector.py` is our reconstruction from the paper.
