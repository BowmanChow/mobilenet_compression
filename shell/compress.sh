compression_path="algorithms/compression"
nets_path="${compression_path}/nets"
model_path="${nets_path}/mbnet_comp"


dataset=$1
model=$2
prune_method=$3
quan_method=$4
ft_lr=$5
ft_bs=$6
ft_epochs=$7
prune_sparisity=$8
gpus=$9
input_path=${10}
output_path=${11}
dataset_path=${12}

source ~/conda3/etc/profile.d/conda.sh
conda activate mbnet

if [ $prune_method != 'null' ] && [ $quan_method == 'null' ] # prune
then
    mkdir -p ${output_path}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python ${model_path}/pruning_experiments.py \
            --input_dir=${input_path} \
            --output_dir=${output_path} \
            --dataset_dir=${dataset_path} \
            --learning_rate=${ft_lr} \
            --finetune_epochs=${ft_epochs} \
            --batch_size=${ft_bs} \
            --calc_initial_yaml \
            --calc_final_yaml \


fi

conda deactivate