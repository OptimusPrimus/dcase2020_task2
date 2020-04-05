## Install
 1. To setup project & download data run the following commands:
    - ```conda env create -f environment.yml```
    - download data (https://zenodo.org/record/3678171#.XnTC7nVKjmE) & unzip into ```~/shared/DCASE2020_Task2```
    
 2. Setup MongoDB & Ominboard for Sacred Logger
    - https://docs.mongodb.com/manual/installation/
    - https://github.com/vivekratnavel/omniboard

## Run experiment

  see scripts folder



### Baseline Results

___



| Machine | Type | ID   | AUC        | pAUC       |
| ------- | :--- | :--- | ---------- | ---------- |
| fan     | 0    | 0    | 0.5639     | 0.4959     |
| fan     | 0    | 2    | 0.8054     | 0.6094     |
| fan     | 0    | 4    | 0.6660     | 0.5412     |
| fan     | 0    | 6    | 0.9021     | 0.7100     |
| slider  | 1    | 0    | 0.6967 (?) | 0.5395 (?) |
| slider  | 1    | 2    | 0.6124 (?) | 0.5770 (?) |
| slider  | 1    | 4    | 0.9497     | 0.7931     |
| slider  | 1    | 6    | 0.8018     | 0.6068     |
| pump    | 2    | 0    | 0.9341     | 0.69914    |
| pump    | 2    | 2    | 0.7753     | 0.6069     |
| pump    | 2    | 4    | 0.9043     | 0.6256 (?) |
| pump    | 2    | 6    | 0.6628 (?) | 0.4979     |
| ToyCar  | 3    | 1    | 0.7975 (-) | 0.6979     |
| ToyCar  | 3    | 2    | 0.8678     | 0.7775     |
| ToyCar  | 3    | 3    | 0.6651     | 0.5633     |
| ToyCar  | 3    | 4    | 0.8859     | 0.7442     |
| ToyCon  | 4    | 1    | 0.7551 (-) | 0.6256 (-) |
| ToyCon  | 4    | 2    | 0.6276 (-) | 0.5589 (-) |
| ToyCon  | 4    | 3    | 0.7386 (-) | 0.5949 (-) |
| valve   | 5    | 0    | 0.6751 (-) | 0.5179     |
| valve   | 5    | 2    | 0.6285 (-) | 0.5105 (-) |
| valve   | 5    | 4    | 0.7339 (-) | 0.5263     |
| valve   | 5    | 6    | 0.5888     | 0.4947     |