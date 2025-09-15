SCENE_DIR="$HOME/data/HDR-Bad-Gaussian/pixel8pro/temp"
#SCENE_LIST="processed_2024_09_17_23_29_10-0"
SCENE_LIST="processed_2024_09_24_13_57_43-262"

RESULT_DIR="results/benchmark_hdr_deblur_phone"
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
        --data_dir $SCENE_DIR/$SCENE/dpvslam_2/ \
        --result_dir $RESULT_DIR/$SCENE/bilagrid_deblur &
    CUDA_VISIBLE_DEVICES=1 python simple_trainer_deblur.py \
        mcmc \
        --camera_optimizer.num_virtual_views 1\
        --disable_viewer \
        --data_factor $DATA_FACTOR \
        --scale_factor 1 \
        --nvs_on_contiguous_images \
        --use_bilateral_grid \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/dpvslam_2/ \
        --result_dir $RESULT_DIR/$SCENE/bilagrid_nodeblur
done

