CUDA_VISIBLE_DEVICES=0 \
python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_17_23_29_10-0/dpvslam_2 \
    --data_factor 1 \
    --scale_factor 1 \
    --exposure_time_lr 1e-2 \
    --result_dir results/pixel8pro/temp/2024_09_17_23_29_10-0_dpvslam_2_random 2>&1 | tee results/pixel8pro/temp/2024_09_17_23_29_10-0_dpvslam_2_random/log.txt &\
CUDA_VISIBLE_DEVICES=1 \
python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_57_43-262/dpvslam_2 \
    --data_factor 1 \
    --scale_factor 1 \
    --exposure_time_lr 1e-2 \
    --result_dir results/pixel8pro/temp/2024_09_24_13_57_43-262_dpvslam_2_random 2>&1 | tee results/pixel8pro/temp/2024_09_24_13_57_43-262_dpvslam_2_random/log.txt

