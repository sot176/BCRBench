import inspect

def _build_model(model_class, **kwargs):
    sig = inspect.signature(model_class.__init__)
    valid_params = sig.parameters.keys()

    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k in valid_params
    }

    return model_class(**filtered_kwargs)


def get_model(model_name: str, **kwargs):

    if model_name == "Mirai":
        from models.Mirai.model import MiraiModel
        return _build_model(MiraiModel, **kwargs)

    elif model_name == "ImgFeatAlign":
        from models.ImgFeatAlign.model import ImgFeatAlign
        return _build_model(ImgFeatAlign, **kwargs)

    elif model_name == "LMV-Net":
        from models.LMVNet.model import LMVNet
        return _build_model(LMVNet, **kwargs)

    elif model_name == "VMRA-MaR":
        from models.VMRAMaR.model import VMRAMaR
        return _build_model(VMRAMaR, **kwargs)

    elif model_name == "OA-BreaCR":
        from models.OABreaCR.model import OABreaCR
        return _build_model(OABreaCR, **kwargs)

    else:
        raise ValueError(f"Unknown model: {model_name}")
