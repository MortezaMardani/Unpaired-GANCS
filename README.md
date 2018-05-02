# Unpaired-GANCS
This code implements the recovery of image x from the undersampled measurements y when the pair (x,y) is not avaialble for training. However, we know the images, say \tilde{x}, from another relevant image dataset that can guide us to recover the ground-truth x from y. This is important for medical imaging applications where usually one doesn't have access to high-resolution datastes for all organs. 

First experiment on my machine (4/24 1807): training process succeeded with Knee data and first version code, killed after 10000 batch

python3 srez_main.py --run train --dataset_train /home/shared/Unpaired-GANCS/Knee-highresolution-19cases/train --dataset_test /home/shared/Unpaired-GANCS/Knee-highresolution-19cases/test  --sample_size 320 --sample_size_y 256 --batch_size 2 --summary_period 10000 --sample_test 200 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 3000 --train_dir /home/shared/Unpaired-GANCS/train_dir/exp1 --gpu_memory_fraction 1.0 --checkpoint_dir ./checkpoint/checkpoint00


First exp second attemp (first failed due to ckpt not restored for training) (4/25 1150): 

python3 srez_main.py --run train --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_5fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 2 --summary_period 5001 --checkpoint_period 5000 --sample_test 100 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 360 --train_dir /home/shared/Unpaired-GANCS/train_dir/exp1  --checkpoint_dir ./checkpoint/checkpoint00


Second exp (4/24 2303): with code commit 7ce731203700ae94241b23051989512b0b1725b4 (i.e. no mse loss for gene but no shuffle), killed after 5000 batch, 400 png saved, no ckpt
python3 srez_main.py --run train --dataset_train /home/shared/Unpaired-GANCS/Knee-highresolution-19cases/train --dataset_test /home/shared/Unpaired-GANCS/Knee-highresolution-19cases/test --sample_size 320 --sample_size_y 256 --batch_size 2 --summary_period 5000 --sample_test -1 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 200 --train_dir /home/shared/Unpaired-GANCS/train_dir/exp2 --gpu_memory_fraction 1.0 


Third exp (4/25 1706): with code commit 9f6ad306a8dd3a2dfcd95525e1b9d47306079ed3, keep pairs by shuffle together
python3 srez_main.py --run train --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 3 --summary_period 20000 --sample_test 100 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 1500 --train_dir ./train_dir/exp3  --checkpoint_period 5000 --checkpoint_dir ./checkpoint/ckpt02


Resume 3rd exp after 25h training (4/26 2037): killed after writing out b5290.png, ckpt at b39710

python3 main.py --run train --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 3 --summary_period 5290 --sample_test 100 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 360 --train_dir ./train_dir/exp3  --checkpoint_period 5000 --checkpoint_dir ./checkpoint/ckpt02 --gpu_memory_fraction 0.95 &


3rd exp version A (4/27 1110): train with pure L1 for first 1500 batches then switch to pure GAN

python3 main.py --run train --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 3 --summary_period 1500 --sample_test 100 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 1500 --train_dir ./train_dir/exp3a --checkpoint_period 1500 --checkpoint_dir ./checkpoint/ckpt02a &


4th exp (4/25 1100): same as 3rd except for breaking pairs for train

python3 srez_main.py --run train --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 3 --summary_period 20000 --sample_test 100 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 1500 --train_dir ./train_dir/mail  --checkpoint_period 5000 --checkpoint_dir ./checkpoint/ckpt03 --permutation_split True


5th exp (4/30 2007): first 100 pure L1, 100~4k 0.54 decay to 0.52, 4k~8k pure GAN

python3 main.py --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 2 --sample_test 50 --train_dir ./train_dir/exp4 --checkpoint_dir ./checkpoint/ckpt04


Given command on group's server: python3 srez_main.py --run train --dataset_train /mnt/raid5/morteza/datasets/Knee-highres-19cases-train-test/train --dataset_test /mnt/raid5/morteza/datasets/Knee-highres-19cases-train-test/test --sampling_pattern /mnt/raid5/morteza/datasets/Knee-highres-19cases-train-test/mask_5fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 2 --summary_period 50000 --sample_test -1 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 3000 --train_dir /mnt/raid5/morteza/train_save_morteza_knee/train_save_all_0.75l2_0.25GAN_5fold_128128128128128gen_1copies --gpu_memory_fraction 1.0 --hybrid_disc 0
