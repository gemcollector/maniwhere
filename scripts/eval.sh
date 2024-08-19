easy_task_list=('ur5_lift')
frames=1001000
feature_dim=256
use_wandb=False
save_snapshot=False
save_video=True
lr_stn=1e-4
img_size=128

# Warning: you should change the model-path in mani_eval.py
for task_name in ${easy_task_list[@]};
do
    CUDA_VISIBLE_DEVICES=7  python mani_eval.py \
                                task=${task_name} \
                                seed=1 \
                                use_wandb=${use_wandb} \
                                img_size=${img_size} \
                                save_snapshot=${save_snapshot}  \
                                lr_stn=${lr_stn} \
                                save_video=${save_video} \
                                wandb_group=$1


done