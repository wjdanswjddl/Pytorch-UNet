# initial arxiv submission
name=plane1
model=/home/abhat/wirecell_sbnd/Pytorch-UNet/checkpoints/UNet/tpc1_bothplanes_with_prolongedtrks/best_loss.pth
time python eval.py -g --model ${model} -o ${name}


# for sample size test
# time python eval.py -g --model sample-size-test-300/CP43.pth -o sample-size-test-300
