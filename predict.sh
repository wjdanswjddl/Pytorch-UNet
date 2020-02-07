# ./predict.py -m l23-cosmic500-e50/CP50-450.pth --viz --no-save --no-crf --input test/data-0.h5 --mask-threshold 0.7 -e 0
# ./predict.py -m model-uresnet-512/CP30-450.pth --viz --no-save --no-crf --input test/data-0.h5 --mask-threshold 0.7 -e 0

./predict.py -m l23-cosmic500-e50/CP50-450.pth --no-save --no-crf --input test/data-0.h5 --mask-threshold -1 -e 0
