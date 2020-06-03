cd ~

git clone git@gitlab.cp.jku.at:paulp/dcase2020_task2.git

cd dcase2020_task2

conda env create -f environment.yml

conda install -c conda-forge librosa
