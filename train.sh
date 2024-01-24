
# model=unet-explr-l23-cosmic500-e60/CP60-450.pth
# model=uresnet-l23-cosmic500-t1/CP20-450.pth

# model=model/unet-l23-cosmic500-e50/CP1-450.pth

# model=unet-adam-l23-cosmic500-e50/CP50-450.pth

#time python train.py -g -e 20 -n 500
time python train.py -g -e 20 -n 500
# time python train.py -g --load ${model} -e 2 -n 10
