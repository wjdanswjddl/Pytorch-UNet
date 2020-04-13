
model=unet-l23-cosmic500-e50-t1/CP41-450.pth

test=/home/yuhw/wc/wct-analysis/ml-sp/backup/eval-dune-collab-2020-01/nf-on/eval-87-85/g4-rec-0.h5
# test=test/data-0.h5
# test=data/cosmic-rec-0.h5

# python predict.py -m ${model} --viz --no-save --no-crf --input ${test} --range 488 489 --mask-threshold 0.7
python predict.py -m ${model} --viz --no-save --no-crf --input ${test} --range 0 1 --mask-threshold 0.5