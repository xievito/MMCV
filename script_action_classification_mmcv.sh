CUDA_VISIBLE_DEVICES=2 python action_classification_mmcv.py \
  --lr 0.01 \
  --view joint \
  --batch-size 128 \
  --pretrained ./checkpoints/pretrain_moco_mmcv/ntu60_cross_subject/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 --protocol cross_subject --finetune-skeleton-representation graph-based


CUDA_VISIBLE_DEVICES=2 python action_classification_mmcv.py \
  --lr 0.01 \
  --view joint \
  --batch-size 128 \
  --pretrained ./checkpoints/pretrain_moco_mmcv/ntu60_cross_view/checkpoint_0450.pth.tar \
  --finetune-dataset ntu60 --protocol cross_subject --finetune-skeleton-representation graph-based
