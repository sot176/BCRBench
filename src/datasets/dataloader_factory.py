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

def get_dataset_and_loader(dataset_name: str, model_name: str, split: str,
                           csv_file: str, data_root: str,
                           batch_size: int, num_workers: int,
                           shuffle: bool, pin_memory: bool,
                           transforms=None, drop_last=False):
    """
    Returns a dataset and dataloader based on dataset_name and model_name.
    Mirrors the get_model factory style.
    """

    # --- Select dataset class based on dataset_name AND model_name ---
    if dataset_name == "CSAW":
        print(f"Using CSAW-CC dataset for {split} split with model {model_name}")
        if model_name == "Mirai":
            dataset_class = BreastCancerRiskDatasetCSAWCC_Mirai
        elif model_name == "ImgFeatAlign":
            dataset_class = BreastCancerRiskDatasetCSAWCC_ImgFeatAlign
        elif model_name == "LMV-Net":
            dataset_class = BreastCancerRiskDatasetCSAWCC_LMVNet
        elif model_name == "VMRA-MaR":
            dataset_class = BreastCancerRiskDatasetCSAWCC_VMRA
        elif model_name == "OA-BreaCR":
            dataset_class = BreastCancerRiskDatasetCSAWCC_ImgFeatAlign

    elif dataset_name == "EMBED":
        print(f"Using EMBED dataset for {split} split with model {model_name}")
        if model_name == "Mirai":
            dataset_class = BreastCancerRiskDatasetEMBED_Mirai
        elif model_name == "ImgFeatAlign":
            dataset_class = BreastCancerRiskDatasetEMBED_ImgFeatAlign
        elif model_name == "LMV-Net":
            dataset_class = BreastCancerRiskDatasetEMBED_LMVNet
        elif model_name == "VMRA-MaR":
            dataset_class = BreastCancerRiskDatasetEMBED_VMRA
        elif model_name == "OA-BreaCR":
            dataset_class = BreastCancerRiskDatasetEMBED_ImgFeatAlign

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Instantiate dataset
    dataset = dataset_class(csv_file, data_root, split, transforms=transforms)

    # --- DataLoader ---
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle if split == "train" else False,
        pin_memory=pin_memory, drop_last=drop_last
    )

    return loader
