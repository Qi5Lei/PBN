#!/bin/bash

cd ..

DATA=/path/data/mixstyle
DASSL=/path/Dassl.pytorch

D1=art_painting
D2=cartoon
D3=photo
D4=sketch

apply_layer_tuple=("0,1,2,3,4")
base_norm=BN
update_norm=StochNorm2d
replace_norm_name=block_split_random_block_bn2d_two_type_channel_fusion_close_affine

############## single source generalization setting.
DATASET=pacs
TRAINER=Vanilla2_pbn
NET=resnet50_pbn
MIX=random

for apply_layer in ${apply_layer_tuple[*]}
do
  for SEED in $(seq 1 5)
  do
      for SETUP in $(seq 1 4)
      do
          if [ ${SETUP} == 1 ]; then
              S1=${D2}
              S2=${D3}
              S3=${D4}
              T=${D1}
          elif [ ${SETUP} == 2 ]; then
              S1=${D1}
              S2=${D3}
              S3=${D4}
              T=${D2}
          elif [ ${SETUP} == 3 ]; then
              S1=${D1}
              S2=${D2}
              S3=${D4}
              T=${D3}
          elif [ ${SETUP} == 4 ]; then
              S1=${D1}
              S2=${D2}
              S3=${D3}
              T=${D4}
          fi

          CUDA_VISIBLE_DEVICES=0,1 python train_pbn.py \
          --origin-norm ${base_norm} --norm-type StochBN --apply-layer $apply_layer \
          --update-norm ${update_norm} --stoch-norm-alpha 0.5 --num-norm 1 --replace-norm True \
          --point-group 4 --replace-norm-name ${replace_norm_name} \
          --root ${DATA} \
          --seed ${SEED} \
          --trainer ${TRAINER} \
          --source-domains ${T}  \
          --target-domains ${S1} ${S2} ${S3} \
          --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
          --config-file configs/trainers/mixstyle/${DATASET}_norm.yaml \
          --output-dir PBNix/${DATASET}/${TRAINER}/${NET}_${base_norm}_${replace_norm_name}_${apply_layer}/${MIX}/${T}/seed${SEED} \
          MODEL.BACKBONE.NAME ${NET}

      done
  done
done