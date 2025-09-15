SCENE_DIR="$HOME/data/HDR-Bad-Gaussian/bags/20240921"
# SCENE_LIST="hdr_store_feat_girls20240921-231313"
SCENE_LIST="hdr_toufu_feat_ltdz20240921-225936"

RESULT_DIR="results/benchmark_hdr_deblur_realsense"
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
        --nvs_eval_start 20 \
        --nvs_eval_end 20 \
        --use_bilateral_grid \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/dpvslam_pruned/process \
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
        --data_dir $SCENE_DIR/$SCENE/dpvslam_pruned/process \
        --result_dir $RESULT_DIR/$SCENE/bilagrid_nodeblur
done

