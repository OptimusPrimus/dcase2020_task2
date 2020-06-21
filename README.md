## Technical Report

See ```Primus_CP-JKU.pdf```.

## Install
 1. To setup project & download data run the following commands:
    - ```./setup.sh```
    - download data (https://zenodo.org/record/3678171#.XnTC7nVKjmE) & unzip into ```~/shared/DCASE2020_Task2``` (you can specify a different location in the configuration)
    
 2. Setup MongoDB & Ominboard for Sacred Logger
    - https://docs.mongodb.com/manual/installation/
    - https://github.com/vivekratnavel/omniboard


## References

- Yuma Koizumi, Yohei Kawaguchi, Keisuke Imoto, Toshiki Nakamura, Yuki Nikaido, Ryo Tanabe, Harsh Purohit, Kaori Suefusa, Takashi Endo, Masahiro Yasuda, and Noboru Harada. *Description and discussion on DCASE2020 challenge task2: unsupervised anomalous sound detection for machine condition monitoring.* In arXiv e-prints: 2006.05822, 1–4. June 2020. URL: https://arxiv.org/abs/2006.05822.
- Yuma Koizumi, Shoichiro Saito, Hisashi Uematsu, Noboru Harada, and Keisuke Imoto. *ToyADMOS: a dataset of miniature-machine operating sounds for anomalous sound detection.* In Proceedings of IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 308–312. November 2019. URL: https://ieeexplore.ieee.org/document/8937164.
- Harsh Purohit, Ryo Tanabe, Takeshi Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi. *MIMII Dataset: sound dataset for malfunctioning industrial machine investigation and inspection.* In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019), 209–213. November 2019. URL: http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Purohit_21.pdf.

## Citation

If you use the model or the model implementation please cite the following paper:
```
@inproceedings{Koutini2019Receptive,
    author      =   {Koutini, Khaled and Eghbal-zadeh, Hamid and Dorfer, Matthias and Widmer, Gerhard},
    title       =   {{The Receptive Field as a Regularizer in Deep Convolutional Neural Networks for Acoustic Scene Classification}},
    booktitle   =   {Proceedings of the European Signal Processing Conference (EUSIPCO)},
    address     =   {A Coru\~{n}a, Spain},
    year        =   2019
}
```
If you use other parts of the implementation please cite:
```
@techreport{Primus2019DCASE,
    Author      =   {Primus, Paul},
    institution =   {{DCASE2020 Challenge}},
    title       =   {Reframing Unsupervised Machine Condition Monitoring as a Supervised Classification Task with Outlier-Exposed Classifiers},
    month       =   {June},
    year        =   2019
}
```

## Links
- [Sacred GitHub](https://github.com/IDSIA/sacred)
- [Omniboard](https://github.com/vivekratnavel/omniboard)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)