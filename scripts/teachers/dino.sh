# Publication title          : Emerging Properties in Self-Supervised Vision Transformers
# Publication URL            : https://arxiv.org/abs/2104.14294
# Official Github repo       : https://github.com/facebookresearch/dino

##################################################
# Code for preparing model(s):
##################################################
# Arguments:
# root_dir: path where all models are saved
root_dir=${1}

for arch in "vitbase16"; do

    if [[ ${arch} == "vitsmall16" ]]; then
        model_url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif [[ ${arch} == "vitbase16" ]]; then
        model_url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    else
        echo "==> Unknown architecture: ${arch}"
        exit
    fi

    echo "Preparing DINO - ${arch}"
    model_dir=${root_dir}/dino/${arch}
    mkdir -p ${model_dir}
    echo "==> Model directory: ${model_dir}"

    cd ${model_dir}
    wget -q -O model.pth ${model_url}
    model_ckpt="${model_dir}/model.pth"
    if [[ ! -f "${model_ckpt}" ]]; then
        echo "==> Couldn't download model checkpoint, please see the bash script for possible further instructions."
    else
        echo "==> Model checkpoint: ${model_ckpt}"
    fi

done