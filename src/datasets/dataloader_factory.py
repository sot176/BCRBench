import inspect
from torch.utils.data import DataLoader


def _build_dataset(dataset_class, **kwargs):
    sig = inspect.signature(dataset_class.__init__)
    valid_params = sig.parameters.keys()

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return dataset_class(**filtered_kwargs)


# ── Lazy dataset builders ────────────────────────────────────────
def _csaw_mirai(**kwargs):
    from .CSAW.dataset_csaw_Mirai import BreastCancerRiskDatasetCSAWMirai
    return _build_dataset(BreastCancerRiskDatasetCSAWMirai, **kwargs)


def _csaw_imgfeatalign(**kwargs):
    from .CSAW.dataset_csaw_ImgFeatAlign import BreastCancerRiskDatasetCSAWCCImgFeatAlign
    return _build_dataset(BreastCancerRiskDatasetCSAWCCImgFeatAlign, **kwargs)


def _csaw_lmvnet(**kwargs):
    from .CSAW.dataset_csaw_LMVNet import BreastCancerRiskDatasetCSAWCCLMVNet
    return _build_dataset(BreastCancerRiskDatasetCSAWCCLMVNet, **kwargs)


def _csaw_vmra(**kwargs):
    from .CSAW.dataset_csaw_VMRA import BreastCancerRiskDatasetCSAWVMRA
    return _build_dataset(BreastCancerRiskDatasetCSAWVMRA, **kwargs)


def _embed_mirai(**kwargs):
    from .EMBED.dataset_embed_Mirai import BreastCancerRiskDatasetEMBEDMirai
    return _build_dataset(BreastCancerRiskDatasetEMBEDMirai, **kwargs)


def _embed_imgfeatalign(**kwargs):
    from .EMBED.dataset_embed_ImgFeatAlign import BreastCancerRiskDatasetEMBEDImgFeatAlign
    return _build_dataset(BreastCancerRiskDatasetEMBEDImgFeatAlign, **kwargs)


def _embed_lmvnet(**kwargs):
    from .EMBED.dataset_embed_LMVNet import BreastCancerRiskDatasetEMBEDLMVNet
    return _build_dataset(BreastCancerRiskDatasetEMBEDLMVNet, **kwargs)


def _embed_vmra(**kwargs):
    from .EMBED.dataset_embed_VMRA import BreastCancerRiskDatasetEMBEDVMRA
    return _build_dataset(BreastCancerRiskDatasetEMBEDVMRA, **kwargs)


# ── Registry ─────────────────────────────────────────────────────
DATASET_REGISTRY = {
    "CSAW": {
        "Mirai": _csaw_mirai,
        "ImgFeatAlign": _csaw_imgfeatalign,
        "LMV-Net": _csaw_lmvnet,
        "VMRA-MaR": _csaw_vmra,
        "OA-BreaCR": _csaw_imgfeatalign,   
    },
    "EMBED": {
        "Mirai": _embed_mirai,
        "ImgFeatAlign": _embed_imgfeatalign,
        "LMV-Net": _embed_lmvnet,
        "VMRA-MaR": _embed_vmra,
        "OA-BreaCR": _embed_imgfeatalign,
    },
}


def get_dataset_and_loader(
    dataset_name: str,
    model_name: str,
    split: str,
    csv_file: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    pin_memory: bool = True,
    transforms=None,
    sampler=None,
    **dataset_kwargs,
):
    """
    Flexible dataset + dataloader factory.
    """

    try:
        dataset_builder = DATASET_REGISTRY[dataset_name][model_name]
    except KeyError:
        raise ValueError(
            f"Unsupported combination: dataset='{dataset_name}', model='{model_name}'. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    print(f"[INFO] Dataset={dataset_name} | Model={model_name} | Split={split}")

    dataset = dataset_builder(
    csv_file=csv_file,
    image_dir=data_root,
    mode=split,
    transforms=transforms,
    **dataset_kwargs
)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(shuffle if split == "train"  else False),
        sampler=sampler,
        pin_memory=pin_memory,
    )

    return loader