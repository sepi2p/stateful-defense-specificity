"""Loader for the CIFAR-10 models of BlackboxBench (reduced copy for this release).

Only the function that the study uses is kept. The model definitions and weights of BlackboxBench
are licensed CC BY-NC 4.0 and are not redistributed here: copy the directory with the CIFAR-10
model definitions of https://github.com/SCLBD/BlackboxBench to `surro_models/blackboxbench_cifar10/`
(it has to export `vgg19_bn`) and its checkpoint to
`checkpoints/blackboxbench_cifar10/ckpt/vgg19_bn/model_best.pth.tar`.
"""

import os

import torch


def _load_state_dict_strict(model, checkpoint_path, state_dict_key='state_dict'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Required checkpoint not found: {checkpoint_path}. "
            "BlackboxBench model names must not fall back to random weights."
        )

    print(f"[INFO] Loading BlackboxBench CIFAR-10 checkpoint from {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
    state_dict = checkpoint[state_dict_key] if state_dict_key in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def load_blackboxbench_cifar_model(
    model_name,
    home_path='checkpoints/blackboxbench_cifar10/ckpt',
    require_optim=False,
):
    from advertorch.utils import NormalizeByChannelMeanStd

    from surro_models.blackboxbench_cifar10 import vgg19_bn as bbb_vgg19_bn

    print('Load BlackboxBench cifar model: ', model_name)

    if model_name == 'bbb_vgg19_bn':
        pretrained_model = bbb_vgg19_bn(num_classes=10)
        model_checkpoint_path = os.path.join(home_path, 'vgg19_bn', 'model_best.pth.tar')
        # BlackboxBench's checkpoint was saved with only the feature extractor wrapped.
        pretrained_model.features = torch.nn.DataParallel(pretrained_model.features)
        _load_state_dict_strict(pretrained_model, model_checkpoint_path)
        pretrained_model.features = pretrained_model.features.module
    else:
        raise NotImplementedError(f"{model_name}: only bbb_vgg19_bn is used in the study")

    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
    model = torch.nn.Sequential(
        normalize,
        pretrained_model
    )

    model = model.cuda()
    model.eval()
    if require_optim:
        optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=1e-4)
        return model, optimizer
    return model
