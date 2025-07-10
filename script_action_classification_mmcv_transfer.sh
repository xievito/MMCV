# Semi with data ratio = 1.0 on PKU-MMD II
for((i=1;i<=20;i++));
do
CUDA_VISIBLE_DEVICES=2 python action_classification_mmcv_semi.py \
  --lr 0.01 \
  --batch-size 128 \
  --pretrained ./checkpoints/pretrain_moco_mmcv/ntu60_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset pku_v2 \
  --protocol cross_subject_semi \
  --data-ratio 1.0 \
  --finetune-skeleton-representation graph-based
done

for((i=1;i<=20;i++));
do
CUDA_VISIBLE_DEVICES=2 python action_classification_mmcv_semi.py \
  --lr 0.01 \
  --batch-size 128 \
  --pretrained ./checkpoints/pretrain_moco_mmcv/ntu60_cross_view/checkpoint_0450.pth.tar \
  --finetune-dataset pku_v2 \
  --protocol cross_subject_semi \
  --data-ratio 1.0 \
  --finetune-skeleton-representation graph-based
done
