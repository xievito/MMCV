
# ntu60-subject
CUDA_VISIBLE_DEVICES=2 python pretrain_moco_mmcv.py \
--lr 0.02 \
--batch-size 128 \
--teacher-t 0.05 \
--student-t 0.1 \
--topk 8192 \
--mlp \
--contrast-t 0.07 \
--contrast-k 16384 \
--checkpoint-path ./checkpoints/pretrain_moco_mmcv/ntu60_cross_subject \
--schedule 351 \
--epochs 451 \
--pre-dataset ntu60 \
--skeleton-representation graph-based \
--protocol cross_subject


# ntu60-view
CUDA_VISIBLE_DEVICES=2 python pretrain_moco_mmcv.py \
--lr 0.02 \
--batch-size 128 \
--teacher-t 0.05 \
--student-t 0.1 \
--topk 8192 \
--mlp \
--contrast-t 0.07 \
--contrast-k 16384 \
--checkpoint-path ./checkpoints/pretrain_moco_mmcv/ntu60_cross_view \
--schedule 351 \
--epochs 451 \
--pre-dataset ntu60 \
--skeleton-representation graph-based \
--protocol cross_view