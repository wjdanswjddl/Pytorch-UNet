name=unet-l23-cosmic500-e50-t2

model=model/${name}/CP50-450.pth

time python eval.py -g --model ${model} -o ${name}