yes | CUDA_VISIBLE_DEVICES=0 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scene0489_02/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/scene0489_02_dpvslam_scale1 &&\
yes | CUDA_VISIBLE_DEVICES=0 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scene0489_02/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/scene0489_02_dpvslam_scale2 &&\
yes | CUDA_VISIBLE_DEVICES=0 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scannet_restored/scene0489_02/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/scannet_restored/scene0489_02_dpvslam_scale1 &&\
yes | CUDA_VISIBLE_DEVICES=0 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scannet_restored/scene0489_02/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/scannet_restored/scene0489_02_dpvslam_scale2 &

yes | CUDA_VISIBLE_DEVICES=1 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scene0077_00/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/scene0077_00_dpvslam_scale1 &&\
yes | CUDA_VISIBLE_DEVICES=1 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scene0077_00/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/scene0077_00_dpvslam_scale2 &&\
yes | CUDA_VISIBLE_DEVICES=1 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scannet_restored/scene0077_00/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/scannet_restored/scene0077_00_dpvslam_scale1 &&\
yes | CUDA_VISIBLE_DEVICES=1 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scannet_restored/scene0077_00/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/scannet_restored/scene0077_00_dpvslam_scale2 &

yes | CUDA_VISIBLE_DEVICES=2 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scene0072_01/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/scene0072_01_dpvslam_scale1 &&\
yes | CUDA_VISIBLE_DEVICES=2 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scene0072_01/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/scene0072_01_dpvslam_scale2 &&\
yes | CUDA_VISIBLE_DEVICES=2 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scannet_restored/scene0072_01/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/scannet_restored/scene0072_01_dpvslam_scale1 &&\
yes | CUDA_VISIBLE_DEVICES=2 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scannet_restored/scene0072_01/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/scannet_restored/scene0072_01_dpvslam_scale2 &

yes | CUDA_VISIBLE_DEVICES=3 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scene0036_00/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/scene0036_00_dpvslam_scale1 &&\
yes | CUDA_VISIBLE_DEVICES=3 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scene0036_00/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/scene0036_00_dpvslam_scale2 &&\
yes | CUDA_VISIBLE_DEVICES=3 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scannet_restored/scene0036_00/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/scannet_restored/scene0036_00_dpvslam_scale1 &&\
yes | CUDA_VISIBLE_DEVICES=3 \
    python simple_trainer_deblur_continuous.py \
    mcmc \
    --disable_viewer \
    --data_dir $HOME/data/HDR-Bad-Gaussian/scannet_restored/scene0036_00/dpvslam \
    --data_factor 2 \
    --scale_factor 1 \
    --result_dir results/scannet_restored/scene0036_00_dpvslam_scale2 &