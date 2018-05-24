# Unpaired-GANCS
This code implements the recovery of image x from the undersampled measurements y when the pair (x,y) is not avaialble for training. However, we know the images, say \tilde{x}, from another relevant image dataset that can guide us to recover the ground-truth x from y. This is important for medical imaging applications where usually one doesn't have access to high-resolution datastes for all organs. 

First experiment on my machine (4/24 1807): training process succeeded with Knee data and first version code, killed after 10000 batch

python3 srez_main.py --run train --dataset_train /home/shared/Unpaired-GANCS/Knee-highresolution-19cases/train --dataset_test /home/shared/Unpaired-GANCS/Knee-highresolution-19cases/test  --sample_size 320 --sample_size_y 256 --batch_size 2 --summary_period 10000 --sample_test 200 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 3000 --train_dir /home/shared/Unpaired-GANCS/train_dir/exp1 --gpu_memory_fraction 1.0 --checkpoint_dir ./checkpoint/checkpoint00


First exp second attemp (first failed due to ckpt not restored for training) (4/25 1150): 

python3 srez_main.py --run train --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_5fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 2 --summary_period 5001 --checkpoint_period 5000 --sample_test 100 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 360 --train_dir /home/shared/Unpaired-GANCS/train_dir/exp1  --checkpoint_dir ./checkpoint/checkpoint00


2nd exp (4/24 2303): with code commit 7ce731203700ae94241b23051989512b0b1725b4 (i.e. no mse loss for gene but no shuffle), killed after 5000 batch, 400 png saved, no ckpt
python3 srez_main.py --run train --dataset_train /home/shared/Unpaired-GANCS/Knee-highresolution-19cases/train --dataset_test /home/shared/Unpaired-GANCS/Knee-highresolution-19cases/test --sample_size 320 --sample_size_y 256 --batch_size 2 --summary_period 5000 --sample_test -1 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 200 --train_dir /home/shared/Unpaired-GANCS/train_dir/exp2 --gpu_memory_fraction 1.0 


Third exp (4/25 1706): with code commit 9f6ad306a8dd3a2dfcd95525e1b9d47306079ed3, keep pairs by shuffle together
python3 srez_main.py --run train --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 3 --summary_period 20000 --sample_test 100 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 1500 --train_dir ./train_dir/exp3  --checkpoint_period 5000 --checkpoint_dir ./checkpoint/ckpt02


Resume 3rd exp after 25h training (4/26 2037): killed after writing out b5290.png, ckpt at b39710

python3 main.py --run train --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 3 --summary_period 5290 --sample_test 100 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 360 --train_dir ./train_dir/exp3  --checkpoint_period 5000 --checkpoint_dir ./checkpoint/ckpt02 --gpu_memory_fraction 0.95 &


3rd exp version A (4/27 1110): train with pure L1 for first 1500 batches then switch to pure GAN

python3 main.py --run train --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 3 --summary_period 1500 --sample_test 100 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 1500 --train_dir ./train_dir/exp3a --checkpoint_period 1500 --checkpoint_dir ./checkpoint/ckpt02a &


3rd exp version B (4/25 1100): same as 3rd except for breaking pairs for train

python3 srez_main.py --run train --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 3 --summary_period 20000 --sample_test 100 --sample_train -1 --subsample_test -1 --subsample_train -1 --train_time 1500 --train_dir ./train_dir/mail  --checkpoint_period 5000 --checkpoint_dir ./checkpoint/ckpt03 --permutation_split True


4th exp (4/30 2007): first 100 pure L1, 100\~4k 0.54 decay to 0.52, 4k\~8k pure GAN

python3 main.py --dataset_train ./Knee-highresolution-19cases/train --dataset_test ./Knee-highresolution-19cases/test --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_320_256_knee_vdrad.mat --sample_size 320 --sample_size_y 256 --batch_size 2 --sample_test 50 --train_dir ./train_dir/exp4 --checkpoint_dir ./checkpoint/ckpt04


5th exp (5/2 1353): same as above except halved image size and feature map size, batch size to 8, ran until 30k

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_160_128_knee_vdrad.mat --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 40 --train_dir ./train_dir/exp5 --checkpoint_dir ./checkpoint/ckpt05


6th exp (5/2 1910): same as above, except first 3k pure L1, 3k~3.2k linear decay to 0, ran until 47k

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_160_128_knee_vdrad.mat --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 40 --train_dir ./train_dir/exp6 --checkpoint_dir ./checkpoint/ckpt06


7th exp (5/4 1635): same as above, except 3.2k+ keep 0.1 mse

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_160_128_knee_vdrad.mat --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 40 --train_dir ./train_dir/exp7 --checkpoint_dir ./checkpoint/ckpt07 --starting_batch 3000


8th exp (5/4 1730): same as above, except no downsampling, 3.2k+ keep 0.8 mse, converged after 8k

9th exp (5/4 1920): same as 7th, except 3.2k+ keep 0.8 mse, converged  after 50k 

10th exp (5/5 2100): same as 6th, except no undersampling, converged after 9k

11th exp (5/7 1510): no dc, unmasked input, 100 mse, pure GAN after 200 

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern nomask --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 24 --train_dir ./train_dir/exp11 --checkpoint_dir ./checkpoint/ckpt11

12th exp (5/8 1710): same as 11th, except 1500 mse, pure GAN after 1700 

13th exp (5/9): same as 12 except 3000 mse, pure GAN after 3200

14th exp (5/11): Added skip connection in gene_model_with_scale, res block reduced from 2 to 1, start with 2000 mse

15th exp (5/17): same as 14, except using 16 patches and start with 3000 mse

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern nomask --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 24 --train_dir ./train_dir/exp15 --checkpoint_dir ./checkpoint/ckpt15 --use_patches True

16th exp (5/18): same as 14, except using 16 patches and crop disc_real_input to 160x100

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern nomask --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 24 --train_dir ./train_dir/exp16 --checkpoint_dir ./checkpoint/ckpt16 --use_patches True --mse_batch 2000

17th exp (5/22): same as 15, except patching for both real and fake input to disc and mse=2000, same command as 16

18th exp (5/22): use 16 patches, 2_fold mask, with DC layer and gene skip connect, larger step size

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_2fold_160_128_knee_vdrad.mat --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 24 --train_dir ./train_dir/exp18 --checkpoint_dir ./checkpoint/ckpt18 --mse_batch 2000 --use_patches True --learning_rate_start 0.0001

19th exp (5/22): same as 18 except mse=100 (same as mse=2000)

20th exp (5/22): same as 19 except pure GAN

--mse_batch -200

20a (5/23): decaying learning rate from 4k (not better, disgard)

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_2fold_160_128_knee_vdrad.mat --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 24 --train_dir ./train_dir/exp20a --checkpoint_dir ./checkpoint/ckpt20a --mse_batch -200 --use_patches True --learning_rate_start 0.00005 --learning_rate_half_life 5000 --starting_batch 4000

21th exp (5/22): same as 20 except no patch (is it just the lr?)

22th exp (5/23): same as 20 except 8 res layers in GEN

22a: remove longest skip connect (worse than with the skip, disgard)
