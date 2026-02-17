def get_model(model_name: str, **kwargs):
    if model_name == "mirai":
        from models.Mirai.model import MiraiModel
        return MiraiModel(**kwargs)

    elif model_name == "ImgFeatAlign":
        from models.ImgFeatAlign.model import ImgFeatAlign
        return ImgFeatAlign(**kwargs)

    elif model_name == "LMV-Net":
        from models.LMVNet.model import LMVNet
        return LMVNet(**kwargs)

    elif model_name == "VMRA-MaR":
        from models.VMRAMaR.model import VMRAMaR
        return VMRAMaR(**kwargs)

    elif model_name == "OA-BreaCR":
        from models.OABreaCR.model import OABreaCR
        return OABreaCR(**kwargs)

    else:
        raise ValueError(f"Unknown model: {model_name}")