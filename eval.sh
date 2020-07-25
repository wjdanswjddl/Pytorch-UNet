name=unet-lt23-cosmic500-e50

model=model/${name}/CP50-450.pth

time python eval.py -g --model ${model} -o ${name}