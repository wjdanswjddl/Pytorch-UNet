
# model=unet-explr-l23-cosmic500-e60/CP60-450.pth
# model=uresnet-l23-cosmic500-t1/CP20-450.pth

# model=model/unet-l23-cosmic500-e50/CP1-450.pth

# model=unet-adam-l23-cosmic500-e50/CP50-450.pth

#time python train2.py -g \
#    --start-epoch 0 --nepoch 50 \
#    --start-train 0 --ntrain 300 \
#    --start-val 450 --nval 50 \
#
time python train2.py -g \
    --start-epoch 0 --nepoch 500 \
    --start-train 0 --ntrain 450\
    --start-val 450 --nval 50 \
    --learning-rate=0.01
