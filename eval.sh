
# initial arxiv submission
name=unet-plane0
#model=checkpoints/best_model_20231005.pth
model=checkpoints/tpc0-bothplanes_shuffle_crop/best_dice.pth
time python eval.py -g --model ${model} -o ${name}


# for sample size test
# time python eval.py -g --model sample-size-test-300/CP43.pth -o sample-size-test-300
