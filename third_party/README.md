# Third-party code (not vendored)

These upstream repositories are cloned locally and ignored by git. Clone them at
the pinned commits to reproduce the stateful-specificity experiments.

| Path | Upstream | Commit | Used for |
|---|---|---|---|
| `third_party/GWAD_official` | https://github.com/jpark04-qub/GWAD | 822b258eb28c4c51eefa8d2cc87072991f750b74 | GWAD / GWAD+ detectors and Delta-Net weights (`model/delta/delta_ann.pt`) |
| `third_party/stateful_monitoring_baselines/blacklight` | https://github.com/Huiying-Li/blacklight | b130337fe3607cb371a42cf2da74459659643f24 | Reference for the Blacklight port (`test_blacklight_port.py`) |
| `third_party/stateful_monitoring_baselines/blackbox-detection` | https://github.com/schoyc/blackbox-detection | 5a3f8017aae54295f4f8c0f79cd9eb43c2ae01e0 | Chen, Carlini & Wagner stateful detection (TensorFlow encoder) |
| `third_party/PyTorch_CIFAR10_official` | https://github.com/huyvnphan/PyTorch_CIFAR10 | 641cac24371b17052b9bb6e56af1c83b5e97cd7f | CIFAR-10 model definitions |
