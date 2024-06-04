
#model=model/unet-l23-cosmic500-e50-t1/CP50-450.pth
#model=checkpoints/best_model_20231005.pth
#model=checkpoints/tpc0-bothplanes_lr1e-1/best_dice.pth
#model=checkpoints/tpc0-bothplanes/best_loss.pth
#model=checkpoints/tpc0-bothplanes/best_loss.pth
#model=checkpoints/tpc0-bothplanes_shuffle/best_dice.pth
#model=checkpoints/tpc0-bothplanes/CP5.pth
#model=checkpoints/tpc0-plane1/best_loss.pth

# model=model/unet-lt-cosmic500-e50/CP50-450.pth
# model=model/unet-explr-l23-cosmic500-e100/CP40-450.pth
# model=model/unet-l23-beam-cosmic500-e100/CP100-450.pth

#best_loss_model
model=/home/abhat/wirecell_sbnd/Pytorch-UNet/checkpoints/tpc0-plane0/Adam/CP30.pth 
model=/home/abhat/wirecell_sbnd/Pytorch-UNet/checkpoints/tpc0-plane1/Adam/best_loss.pth

model=/home/abhat/wirecell_sbnd/Pytorch-UNet/checkpoints/tpc0-plane1/UResNet/Adam/best_loss.pth


# test=/home/yuhw/wc/wct-analysis/ml-sp/backup/eval-dune-collab-2020-01/nf-on/eval-87-85/g4-rec-0.h5
# test=/home/yuhw/wc/wct-analysis/ml-sp/backup/eval-dune-collab-2020-01/nf-on/eval-87-87/g4-rec-0.h5
# test=eval-jinst-resp/eval-87-87/g4-rec-0.h5
# test=test/data-0.h5
# test=data/cosmic-rec-0.h5
#test=data/cosmic-rec-0-v2-test.h5
# test=data/beam-cosmic-rec-0.h5
#test=eval/eval-test/eval-rec-0.h5
#test=eval/eval-thetaxz/eval_80_90/g4-rec-0.h5
#test=/scratch/7DayLifetime/munjung/DNN_ROI/samples/large_angle_tracks_80_90-rec-0_test.h5

#test=/scratch/7DayLifetime/munjung/DNN_ROI/eval/80-82-rec-tpc0_plane0.h5

#test=/scratch/7DayLifetime/munjung/DNN_ROI/train/nu-rec-tpc0_plane0.h5

#test=/scratch/7DayLifetime/munjung/DNN_ROI/train/nu-rec-tpc0_plane1.h5 

test=/scratch/7DayLifetime/munjung/DNN_ROI/eval/85-87-rec-tpc0_plane1.h5
#test=/scratch/7DayLifetime/munjung/DNN_ROI/nu-rec-tpc0_plane1.h5

#test=data/nu-rec-0.h5

# 456, 488, 490, 497
python predict.py -m ${model} --viz --no-crf --input ${test} --range 1 10 --mask-threshold 0.5
