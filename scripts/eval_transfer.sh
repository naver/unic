#!/bin/bash

source ./scripts/setup_env.sh

##########
# extract features
dataset="cars196"
image_size=224
token_type="cls"
pretrained="/path/to/unic/checkpoint.pth"

features_dir=$(dirname "${pretrained}")
features_dir=${features_dir}/transfer/features_${dataset}_${image_size}_${token_type}

if [ ! -f "${features_dir}/features_trainval.pth" ] || [ ! -f "${features_dir}/features_test.pth" ]; then
    echo "Extracting features..."
    python eval_transfer/main_ft_extract.py \
        --output_dir="${features_dir}" \
        --pretrained="${pretrained}" \
        --dataset="${dataset}" \
        --image_size="${image_size}" \
        --token_type="${token_type}"
fi

##########
# train logreg classifier using extracted features
features_norm="none"
clf_type="logreg_sklearn"
if [[ "${dataset}" == "in1k" ]] || [[ "${dataset}" == cog_* ]] || [[ "${dataset}" == inat* ]]; then
    # for large datasets,
    # we use SGD implemented in PyTorch and l2 normalize features
    features_norm="l2"
    clf_type="logreg_torch"
fi

echo ""
echo "Training classifier ..."
python -m sklearnex eval_transfer/main_clf.py --features_dir="${features_dir}" --features_norm=${features_norm} --clf_type=${clf_type}