MODELS_ROOT_DIR="" # set the path to the directory where you want to save the models

if [ -z ${MODELS_ROOT_DIR} ]; then
    echo "Please set the MODELS_ROOT_DIR variable in this script."
    exit -1
fi

umask 000

bash deit-iii.sh ${MODELS_ROOT_DIR}
bash dino.sh ${MODELS_ROOT_DIR}
bash dbot.sh ${MODELS_ROOT_DIR}
bash ibot.sh ${MODELS_ROOT_DIR}
