# Pytorch-UNet For LArTPC Signal Processing

Original Repository: [here](https://github.com/milesial/Pytorch-UNet)

Example usage in `train.sh` and `predict.sh`

## install

prerequisite conda
https://www.anaconda.com/download/

```
conda env create -f conda_env.yml
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
