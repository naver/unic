# Publication title          : DeiT III: Revenge of the ViT
# Publication URL            : https://arxiv.org/abs/2204.07118
# Official Github repo       : https://github.com/facebookresearch/deit/blob/main/README_revenge.md

##################################################
# Code for preparing model(s):
##################################################
# Arguments:
# root_dir: path where all models are saved
root_dir=${1}

for arch in "vitbase16"; do

    if [[ ${arch} == "vitsmall16" ]]; then
        model_url="https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth"
    elif [[ ${arch} == "vitbase16" ]]; then
        model_url="https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth"
    else
        echo "==> Unknown architecture: ${arch}"
        exit
    fi

    echo "Preparing DeiT III - ${arch} ..."
    model_dir=${root_dir}/deit-iii/${arch}
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