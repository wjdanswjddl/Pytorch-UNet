
model=model/unet-l23-cosmic500-e50-t1/CP50-450.pth
model=model/unet-lt-cosmic500-e50/CP50-450.pth
# model=model/unet-explr-l23-cosmic500-e100/CP40-450.pth
# model=model/unet-l23-beam-cosmic500-e100/CP100-450.pth

test=/home/yuhw/wc/wct-analysis/ml-sp/backup/eval-dune-collab-2020-01/nf-on/eval-87-85/g4-rec-0.h5
# test=/home/yuhw/wc/wct-analysis/ml-sp/backup/eval-dune-collab-2020-01/nf-on/eval-87-87/g4-rec-0.h5
# test=test/data-0.h5
# test=data/cosmic-rec-0.h5
test=data/beam-cosmic-rec-0.h5

# 456, 488, 490, 497
python predict.py -m ${model} --viz --no-save --no-crf --input ${test} --range 65 100 --mask-threshold 0.5