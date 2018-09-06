# Unpaired-GANCS

This code implements the recovery of image x from the undersampled measurements y when the pair (x,y) is not avaialble for training. However, we have a small amount of ground-truth x from y. This is important for medical imaging applications where usually one doesn't have access to high-resolution datastes for all organs. 

-------------------------------------------------------------------
exp29 (7/28 2112): wgan 1/3 label, 5-fold, else same as exp28

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_label ./Knee-highresolution-19cases/partial_labels --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_5fold_160_128_knee_vdrad.mat --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 24 --summary_period 1700 --train_dir ./train_dir/exp29 --checkpoint_dir ./checkpoint/exp29 --mse_batch -200 --wgan_gp True --activation lrelu --learning_rate_start 5e-5


compare6 (7/29 2200): 1/3 label and INPUT, else same as exp28

python3 main.py --dataset_train ./Knee-highresolution-19cases/partial_labels --dataset_label ./Knee-highresolution-19cases/partial_labels --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_3fold_160_128_knee_vdrad.mat --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 24 --summary_period 1700 --train_dir ./train_dir/compare6 --checkpoint_dir ./checkpoint/compare6 --mse_batch -200 --wgan_gp True --activation lrelu --learning_rate_start 5e-5

exp30 (8/1 2033): 1/2 label (9 pat.), else same as exp29

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_label ./Knee-highresolution-19cases/partial_labels2 ...

-exp31 (8/2 1700): feature matching first try

-exp31a (8/2 2230): feature matching 1/3 label with commit c2cac4d

-exp32 (8/5 1600): FM with LSGAN no patch, 1/3 label. MSE diverged

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_label ./Knee-highresolution-19cases/partial_labels --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_5fold_160_128_knee_vdrad.mat --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 24 --summary_period 1000 --train_dir ./train_dir/exp32 --checkpoint_dir ./checkpoint/exp32 --mse_batch -200 --activation lrelu --learning_rate_start 1e-4 --FM True

exp33 (8/5 1624): FM plus wgan loss for gene, wgan for disc, with commit 03c8779

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_label ./Knee-highresolution-19cases/partial_labels --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_5fold_160_128_knee_vdrad.mat --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 24 --summary_period 1700 --train_dir ./train_dir/exp33 --checkpoint_dir ./checkpoint/exp33 --mse_batch -200 --wgan_gp True --activation lrelu --learning_rate_start 5e-5 --FM True

exp34 (8/9 2100): wgan with patch, commit 919f651

python3 main.py --dataset_train ./Knee-highresolution-19cases/train_small --dataset_label ./Knee-highresolution-19cases/partial_labels --dataset_test ./Knee-highresolution-19cases/test_small --sampling_pattern ./Knee-highresolution-19cases/sampling_pattern/mask_5fold_160_128_knee_vdrad.mat --sample_size 160 --sample_size_y 128 --batch_size 8 --sample_test 24 --summary_period 1700 --train_dir ./train_dir/exp34 --checkpoint_dir ./checkpoint/exp34 --mse_batch -200 --wgan_gp True --activation lrelu --learning_rate_start 5e-5 --use_patches True &



TODO: stochastic label, dropout/Gaussian in input/intermidiate disc layers, pretrain gene with pure mse and ground truth input with data outside train set
