yes | CUDA_VISIBLE_DEVICES=0 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_28_45-0/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/pixel8pro/2024_09_17_23_28_45 &&\
yes | CUDA_VISIBLE_DEVICES=0 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_28_45-0/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/pixel8pro/2024_09_17_23_28_45_scale2 &

yes | CUDA_VISIBLE_DEVICES=1 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_29_10-0/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/pixel8pro/2024_09_17_23_29_10 &&\
yes | CUDA_VISIBLE_DEVICES=1 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_29_10-0/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/pixel8pro/2024_09_17_23_29_10_scale2 &

yes | CUDA_VISIBLE_DEVICES=2 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_30_22-0/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/pixel8pro/2024_09_17_23_30_22 &&\
yes | CUDA_VISIBLE_DEVICES=2 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_30_22-0/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/pixel8pro/2024_09_17_23_30_22_scale2 &

yes | CUDA_VISIBLE_DEVICES=3 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_36_23-0/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/pixel8pro/2024_09_17_23_36_23 &&\
yes | CUDA_VISIBLE_DEVICES=3 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_36_23-0/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/pixel8pro/2024_09_17_23_36_23_scale2 &
