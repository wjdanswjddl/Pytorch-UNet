
# initial arxiv submission
name=unet-l23-cosmic500-e50-t1
name=unet-lt-cosmic500-e50
model=model/${name}/CP50-450.pth
time python eval.py -g --model ${model} -o ${name}-u


# for sample size test
# time python eval.py -g --model sample-size-test-300/CP43.pth -o sample-size-test-300