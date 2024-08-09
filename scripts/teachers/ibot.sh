# Publication title          : iBOT: Image BERT Pre-Training with Online Tokenizer
# Publication URL            : https://arxiv.org/abs/2111.07832
# Official Github repo       : https://github.com/bytedance/ibot

##################################################
# Code for preparing model(s):
##################################################
# Arguments:
# root_dir: path where all models are saved
root_dir=${1}

for arch in "vitbase16";  do

    if [[ ${arch} == "vitbase16" ]]; then
        # link for the pretrained model
        model_url="https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/checkpoint_teacher.pth"
    else
        echo "==> Unknown architecture: ${arch}"
        exit
    fi

    echo "Preparing iBOT - ${arch}"
    model_dir=${root_dir}/ibot/${arch}
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