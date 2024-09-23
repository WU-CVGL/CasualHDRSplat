SCENE_DIR="$HOME/data/HDR-Bad-Gaussian/"
SCENE_LIST="factory_no_background outdoorpool tanabata_new cozyroom_new"

RESULT_DIR="results/benchmark_hdr_deblur_synthetic"
RENDER_TRAJ_PATH="spiral"
DATA_FACTOR=1

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer_deblur.py \
        mcmc \
        --disable_viewer \
        --data_factor $DATA_FACTOR \
        --scale_factor 1 \
        --nvs_on_contiguous_images \
        --use_bilateral_grid \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/bilagrid_deblur1 &
    CUDA_VISIBLE_DEVICES=1 python simple_trainer_deblur.py \
        mcmc \
        --camera_optimizer.num_virtual_views 1\
        --disable_viewer \
        --data_factor $DATA_FACTOR \
        --scale_factor 1 \
        --nvs_on_contiguous_images \
        --use_bilateral_grid \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/bilagrid_nodeblur1
done
