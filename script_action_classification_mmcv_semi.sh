# Cross-subject
for((i=1;i<=5;i++));
do  
CUDA_VISIBLE_DEVICES=2 python action_classification_mmcv_semi.py \
  --lr 0.01 \
  --batch-size 128 \
  --view joint \
  --pretrained ./checkpoints/pretrain_moco_mmcv/ntu60_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_subject_semi \
  --data-ratio 0.01 \
  --finetune-skeleton-representation graph-based
done

# Cross-view
for((i=1;i<=5;i++));
do
CUDA_VISIBLE_DEVICES=2 python action_classification_mmcv_semi.py \
  --lr 0.01 \
  --batch-size 128 \
  --view joint \
  --pretrained ./checkpoints/pretrain_moco_mmcv/ntu60_cross_view/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_view_semi \
  --data-ratio 0.01 \
  --finetune-skeleton-representation graph-based
done
