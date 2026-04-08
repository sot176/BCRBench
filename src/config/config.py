# config/config.py

from pathlib import Path

cfg = {
    "paths": {
        "asymMirai_master_onconet": Path(
            "/pfs/lustrep4/scratch/project_465002861/thrunsol/Risk_prediction_lumi/src/AsymMirai_master/onconet"
        ),

        "mirai_path": Path(
            "/scratch/project_465002861/thrunsol/mirai_pretrained_backbone/mgh_mammo_MIRAI_Base_May20_2019.p"
        ),

        "csaw_path_saved_reg_model": Path(
            "/scratch/project_465002861/thrunsol/Trained_image_registration_models/csawcc/best_model_registration_id-1.pth"
        ),

        "embed_path_saved_reg_model": Path(
            "/scratch/project_465002861/thrunsol/Trained_image_registration_models/embed/best_model_registration_id-1.pth"
        ),
    }
}