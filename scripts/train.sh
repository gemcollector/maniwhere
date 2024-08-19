task_name='ur5_lift'
frames=801000
feature_dim=256
aux_lr=8e-5
use_wandb=False
save_snapshot=True
save_video=False
lr_stn=1e-4
use_traj=False


CUDA_VISIBLE_DEVICES=0  python camera_train.py \
                            task=${task_name} \
                            seed=1 \
                            use_wandb=${use_wandb} \
                            num_train_frames=${frames} \
                            save_snapshot=${save_snapshot}  \
                            agent.aux_l2_coef=200 \
                            agent.aux_tcc_coef=0 \
                            agent.temp=0.1 \
                            agent.aux_coef=500 \
                            agent.aux_latency=200000 \
                            use_traj=${use_traj} \
                            lr_stn=${lr_stn} \
                            wandb_group=$1