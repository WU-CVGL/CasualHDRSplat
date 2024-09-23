touch "/datasets/HDR-Bad-Gaussian/bags/failed_hdr_store20240920-020335/dpvslam/hold=9999"
touch "/datasets/HDR-Bad-Gaussian/bags/failed_hdr_store20240920-020335/dpvslam/k_times=1"
touch "/datasets/HDR-Bad-Gaussian/bags/failed_hdr_store20240920-020039/dpvslam/hold=9999"
touch "/datasets/HDR-Bad-Gaussian/bags/failed_hdr_store20240920-020039/dpvslam/k_times=1"
touch "/datasets/HDR-Bad-Gaussian/bags/failed_hdr_store20240920-020452/dpvslam/hold=9999"
touch "/datasets/HDR-Bad-Gaussian/bags/failed_hdr_store20240920-020452/dpvslam/k_times=1"
touch "/datasets/HDR-Bad-Gaussian/bags/hdr_store20240920-020214/dpvslam/hold=9999"
touch "/datasets/HDR-Bad-Gaussian/bags/hdr_store20240920-020214/dpvslam/k_times=1"

yes | CUDA_VISIBLE_DEVICES=0 \
   python simple_trainer_deblur_continuous.py \
   mcmc \
   --disable_viewer \
   --data_dir /datasets/HDR-Bad-Gaussian/bags/failed_hdr_store20240920-020335/dpvslam \
   --data_factor 2 \
   --scale_factor 1 \
   --result_dir results/pixel8pro/hdr_store20240920-020335_dpvslam

yes | CUDA_VISIBLE_DEVICES=0 \
    python simple_trainer_deblur_continuous.py \
    default \
    --disable_viewer \
    --data_dir /datasets/HDR-Bad-Gaussian/bags/failed_hdr_store20240920-020039/dpvslam \
    --data_factor 1 \
    --scale_factor 1 \
    --result_dir results/pixel8pro/hdr_store20240920-020039_dpvslam_scale1_default

yes | CUDA_VISIBLE_DEVICES=0 \
   python simple_trainer_deblur_continuous.py \
   mcmc \
   --disable_viewer \
   --data_dir /datasets/HDR-Bad-Gaussian/bags/failed_hdr_store20240920-020452/dpvslam \
   --data_factor 2 \
   --scale_factor 1 \
   --result_dir results/pixel8pro/hdr_store20240920-020452_dpvslam

yes | CUDA_VISIBLE_DEVICES=0 \
   python simple_trainer_deblur_continuous.py \
   mcmc \
   --disable_viewer \
   --data_dir /datasets/HDR-Bad-Gaussian/bags/hdr_store20240920-020214/dpvslam \
   --data_factor 2 \
   --scale_factor 1 \
   --result_dir results/pixel8pro/hdr_store20240920-020214_dpvslam

yes | CUDA_VISIBLE_DEVICES=0 \
   python simple_trainer_deblur_continuous.py \
   mcmc \
   --disable_viewer \
   --data_dir /datasets/HDR-Bad-Gaussian/bags/failed_hdr_store20240920-020335/dpvslam \
   --data_factor 1 \
   --scale_factor 1 \
   --result_dir results/pixel8pro/hdr_store20240920-020335_dpvslam_scale1

yes | CUDA_VISIBLE_DEVICES=0 \
   python simple_trainer_deblur_continuous.py \
   mcmc \
   --disable_viewer \
   --data_dir /datasets/HDR-Bad-Gaussian/bags/hdr_toufu_feat_ltdz20240920-012300/dpvslam \
   --data_factor 1 \
   --scale_factor 1 \
   --result_dir results/pixel8pro/hdr_toufu_feat_ltdz20240920-012300_dpvslam_scale1

yes | CUDA_VISIBLE_DEVICES=0 \
   python simple_trainer_deblur_continuous.py \
   mcmc \
   --disable_viewer \
   --data_dir /datasets/HDR-Bad-Gaussian/bags/failed_hdr_fish_feat_girls20240920-014751/dpvslam \
   --data_factor 1 \
   --scale_factor 1 \
   --result_dir results/pixel8pro/dr_fish_feat_girls20240920-014751_dpvslam_scale1
