from torch.utils.data import DataLoader

# CSAW
from .CSAW.dataset_csaw_Mirai import BreastCancerRiskDatasetCSAWCC_Mirai
from .CSAW.dataset_csaw_ImgFeatAlign import BreastCancerRiskDatasetCSAWCC_ImgFeatAlign
from .CSAW.dataset_csaw_LMVNet import BreastCancerRiskDatasetCSAWCC_LMVNet
from .CSAW.dataset_csaw_VMRA import BreastCancerRiskDatasetCSAWCC_VMRA

# EMBED
from .EMBED.dataset_embed_Mirai import BreastCancerRiskDatasetEMBED_Mirai
from .EMBED.dataset_embed_ImgFeatAlign import BreastCancerRiskDatasetEMBED_ImgFeatAlign
from .EMBED.dataset_embed_LMVNet import BreastCancerRiskDatasetEMBED_LMVNet
from .EMBED.dataset_embed_VMRA import BreastCancerRiskDatasetEMBED_VMRA


# --- Dataset registry ---
DATASET_REGISTRY = {
    "CSAW": {
        "Mirai": BreastCancerRiskDatasetCSAWCC_Mirai,
        "ImgFeatAlign": BreastCancerRiskDatasetCSAWCC_ImgFeatAlign,
        "LMV-Net": BreastCancerRiskDatasetCSAWCC_LMVNet,
        "VMRA-MaR": BreastCancerRiskDatasetCSAWCC_VMRA,
        "OA-BreaCR": BreastCancerRiskDatasetCSAWCC_ImgFeatAlign,
    },
    "EMBED": {
        "Mirai": BreastCancerRiskDatasetEMBED_Mirai,
        "ImgFeatAlign": BreastCancerRiskDatasetEMBED_ImgFeatAlign,
        "LMV-Net": BreastCancerRiskDatasetEMBED_LMVNet,
        "VMRA-MaR": BreastCancerRiskDatasetEMBED_VMRA,
        "OA-BreaCR": BreastCancerRiskDatasetEMBED_ImgFeatAlign,
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
    shuffle: bool,
    pin_memory: bool,
    transforms=None,
):
    """
    Factory function to create dataset and dataloader.
    """

    try:
        dataset_class = DATASET_REGISTRY[dataset_name][model_name]
    except KeyError:
        raise ValueError(
            f"Unsupported combination: dataset='{dataset_name}', model='{model_name}'"
        )

    print(f"Using {dataset_name} dataset for {split} split with model {model_name}")

    dataset = dataset_class(
        csv_file=csv_file,
        data_root=data_root,
        split=split,
        transforms=transforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle if split == "train" else False,
        pin_memory=pin_memory,
    )

    return loader