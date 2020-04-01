
# model=checkpoints/CP1-9.pth

# model=model/unet-lt-cosmic500-e50/CP50-450.pth
# model=model/nestedunet-lt-cosmic500-e50/CP50-450.pth

model=model/unet-l23-cosmic500-e50/CP50-450.pth
# model=model/uresnet-512-l23-cosmic500-e50/CP50-450.pth
# model=model/nestedunet-l23-cosmic500-e50/CP50-450.pth

test=test/data-0.h5
# test=data/cosmic-rec-0.h5

# python predict.py -m ${model} --viz --no-save --no-crf --input ${test} --range 488 489 --mask-threshold 0.7
python predict.py -m ${model} --viz --no-save --no-crf --input ${test} --range 1 10 --mask-threshold 0.7