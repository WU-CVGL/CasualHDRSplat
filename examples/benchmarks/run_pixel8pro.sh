# yes | CUDA_VISIBLE_DEVICES=0 \
#     python simple_trainer_deblur_continuous.py \
#     mcmc \
#     --disable_viewer \
#     --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_28_45-0/dpvslam \
#     --data_factor 1 \
#     --scale_factor 1 \
#     --result_dir results/pixel8pro/2024_09_17_23_28_45 &&\
# yes | CUDA_VISIBLE_DEVICES=0 \
#     python simple_trainer_deblur_continuous.py \
#     mcmc \
#     --disable_viewer \
#     --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_28_45-0/dpvslam \
#     --data_factor 2 \
#     --scale_factor 1 \
#     --result_dir results/pixel8pro/2024_09_17_23_28_45_scale2 &

# yes | CUDA_VISIBLE_DEVICES=1 \
#     python simple_trainer_deblur_continuous.py \
#     mcmc \
#     --disable_viewer \
#     --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_29_10-0/dpvslam \
#     --data_factor 1 \
#     --scale_factor 1 \
#     --result_dir results/pixel8pro/2024_09_17_23_29_10 &&\
# yes | CUDA_VISIBLE_DEVICES=1 \
#     python simple_trainer_deblur_continuous.py \
#     mcmc \
#     --disable_viewer \
#     --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_29_10-0/dpvslam \
#     --data_factor 2 \
#     --scale_factor 1 \
#     --result_dir results/pixel8pro/2024_09_17_23_29_10_scale2 &

# yes | CUDA_VISIBLE_DEVICES=2 \
#     python simple_trainer_deblur_continuous.py \
#     mcmc \
#     --disable_viewer \
#     --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_30_22-0/dpvslam \
#     --data_factor 1 \
#     --scale_factor 1 \
#     --result_dir results/pixel8pro/2024_09_17_23_30_22 &&\
# yes | CUDA_VISIBLE_DEVICES=2 \
#     python simple_trainer_deblur_continuous.py \
#     mcmc \
#     --disable_viewer \
#     --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_30_22-0/dpvslam \
#     --data_factor 2 \
#     --scale_factor 1 \
#     --result_dir results/pixel8pro/2024_09_17_23_30_22_scale2 &

# yes | CUDA_VISIBLE_DEVICES=3 \
#     python simple_trainer_deblur_continuous.py \
#     mcmc \
#     --disable_viewer \
#     --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_36_23-0/dpvslam \
#     --data_factor 1 \
#     --scale_factor 1 \
#     --result_dir results/pixel8pro/2024_09_17_23_36_23 &&\
# yes | CUDA_VISIBLE_DEVICES=3 \
#     python simple_trainer_deblur_continuous.py \
#     mcmc \
#     --disable_viewer \
#     --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/processed_2024_09_17_23_36_23-0/dpvslam \
#     --data_factor 2 \
#     --scale_factor 1 \
#     --result_dir results/pixel8pro/2024_09_17_23_36_23_scale2 &

####################################################################################################################

# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_17_23_29_10-0/dpvslam_2/images_test_2
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_40_25-627/dpvslam_2/images_test_2
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_55_25-344/dpvslam_2/images_test_2
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_57_43-262/dpvslam_2/images_test_2
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_17_23_29_10-0/dpvslam_2/images_2
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_40_25-627/dpvslam_2/images_2
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_55_25-344/dpvslam_2/images_2
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_57_43-262/dpvslam_2/images_2
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_17_23_29_10-0/dpvslam_2/images_test
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_40_25-627/dpvslam_2/images_test
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_55_25-344/dpvslam_2/images_test
# rm -rf $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_57_43-262/dpvslam_2/images_test
# cp -r $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_17_23_29_10-0/dpvslam_2/images     $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_17_23_29_10-0/dpvslam_2/images_test
# cp -r $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_40_25-627/dpvslam_2/images $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_40_25-627/dpvslam_2/images_test
# cp -r $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_55_25-344/dpvslam_2/images $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_55_25-344/dpvslam_2/images_test
# cp -r $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_57_43-262/dpvslam_2/images $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_57_43-262/dpvslam_2/images_test

CUDA_VISIBLE_DEVICES=0 \
python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --no-pin-memory \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_17_23_29_10-0/dpvslam_2 \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir $HOME/data/HDR-Bad-Gaussian/code/SCI-gsplat/examples/results/pixel8pro/temp/processed_2024_09_17_23_29_10-0_dpvslam_2 &\
CUDA_VISIBLE_DEVICES=1 \
python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --no-pin-memory \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_40_25-627/dpvslam_2 \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir $HOME/data/HDR-Bad-Gaussian/code/SCI-gsplat/examples/results/pixel8pro/processed_2024_09_24_13_40_25-627_dpvslam_2 &\
CUDA_VISIBLE_DEVICES=2 \
python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --no-pin-memory \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_55_25-344/dpvslam_2 \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir $HOME/data/HDR-Bad-Gaussian/code/SCI-gsplat/examples/results/pixel8pro/temp/processed_2024_09_24_13_55_25-344_dpvslam_2 &\
CUDA_VISIBLE_DEVICES=3 \
python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --no-pin-memory \
    --data_dir $HOME/data/HDR-Bad-Gaussian/pixel8pro/temp/processed_2024_09_24_13_57_43-262/dpvslam_2 \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir $HOME/data/HDR-Bad-Gaussian/code/SCI-gsplat/examples/results/pixel8pro/temp/processed_2024_09_24_13_57_43-262_dpvslam_2 &\
