# Publication title          : Exploring Target Representations for Masked Autoencoders
# Publication URL            : https://arxiv.org/abs/2209.03917
# Official Github repo       : https://github.com/liuxingbin/dbot

##################################################
# Code for preparing model(s):
##################################################
# Arguments:
# root_dir: path where all models are saved
root_dir=${1}

for arch in "vitbase16_ft_in1k";  do

    if [[ ${arch} == "vitbase16" ]]; then
        # link for the pretrained model
        model_url="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/mmodal/dbot/84.5_dbot_base_pre.pth"
    elif [[ ${arch} == "vitbase16_ft_in1k" ]]; then
        model_url="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/mmodal/dbot/84.5_dbot_base_finetune.pth"
    else
        echo "==> Unknown architecture: ${arch}"
        exit
    fi

    echo "Preparing dBOT - ${arch}"
    model_dir=${root_dir}/dbot/${arch}
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