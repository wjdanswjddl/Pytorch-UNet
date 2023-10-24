# Pytorch-UNet For LArTPC Signal Processing

Original Repository: [here](https://github.com/milesial/Pytorch-UNet)

Example usage in `train.sh` and `predict.sh`

## install

### use `conda`
prerequisite: conda
https://docs.anaconda.com/free/anaconda/install/linux/

use environment.yml
```
conda env create -f environment.yml
```

manually
```
conda create --name pt110 python=3.9 numpy
conda activate pt110
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install matplotlib
pip install h5py
```

### use `pip`

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## talks:
 - [ProtoDUNE DRA Meeting, Jan. 08 2020](https://indico.fnal.gov/event/22795/contribution/4)
 - [DUNE Collaboration Meeting, Jan. 29 2020](https://indico.fnal.gov/event/20144/session/8/contribution/98)

## notes
```
h5dump-shared -n data/g4-rec-r9.h5
./scripts/h5plot.py data/g4-rec-r9.h5 /100/frame_loose_lf0
./scripts/h5plot.py data/g4-rec-r9.h5 /100/frame_mp3_roi0
./scripts/h5plot.py data/g4-rec-r9.h5 /100/frame_mp2_roi0
./scripts/h5plot.py data/g4-tru-r9.h5 /103/frame_ductor0
./scripts/h5plot.py data/g4-rec-r9.h5 /103/frame_gauss0
./train3.sh
python plot_epoch.py 1
./to-ts.py -m test0/CP49.pth
```
