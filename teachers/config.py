from .vit_dino import dino_vitbase
from .vit_deit3 import deit3_vitbase
from .vit_dbotft import dbotft_vitbase

TEACHER_CFG = {
    "dino_vitbase_16": {
        "loader": dino_vitbase,
        "ckpt_path": "",
        "ckpt_key": "model",
        "num_features": 768,
        "resolution": 224,
    },
    "deit3_vitbase_16": {
        "loader": deit3_vitbase,
        "ckpt_path": "",
        "ckpt_key": "model",
        "num_features": 768,
        "resolution": 224,
    },
    "ibot_vitbase_16": {
        "loader": dino_vitbase,
        "ckpt_path": "",
        "ckpt_key": "state_dict",
        "num_features": 768,
        "resolution": 224,
    },
    "dbotft_vitbase_16": {
        "loader": dbotft_vitbase,
        "ckpt_path": "",
        "ckpt_key": "model",
        "num_features": 768,
        "resolution": 224,
    },
    "metaclip_vithuge_14": {
        "loader": None,
        "ckpt_path": "",
        "ckpt_key": "",
        "num_features": 1280,
        "resolution": 224,
    },
    "dino2_vitgiant_14": {
        "loader": None,
        "ckpt_path": "",
        "ckpt_key": "",
        "num_features": 1536,
        "resolution": 518,
    },
}