## Install
 1. To setup project & download data run the following commands:
    - ```conda env create -f environment.yml```
    - ```cd raw_data```
    - ```download_data.sh```
 2. Setup MongoDB & Ominboard for Sacred Logger
    - https://docs.mongodb.com/manual/installation/
    - https://github.com/vivekratnavel/omniboard
## Run experiment

python -m experiments.vae_experiment  with 

```
cd vae_priors

# Vanilla
python -m experiments.vae_experiment with -p prior_class=priors.StandardNormalPrior prior.kwargs.weight=1.0 -m student2.cp.jku.at:27017:better_priors 
# Beta VAE
python -m experiments.vae_experiment with -p prior_class=priors.StandardNormalPrior prior.kwargs.weight=150.0 -m student2.cp.jku.at:27017:better_priors 
# Annealed VAE
python -m experiments.vae_experiment with -p prior_class=priors.StandardNormalPrior prior.kwargs.c_max=10 prior.kwargs.c_stop_epoch=200 prior.kwargs.weight=1000.0 -m student2.cp.jku.at:27017:better_priors 
# Factor VAE
python -m experiments.vae_experiment with -p prior_class=priors.StandardNormalPrior prior.kwargs.weight=1.0 use_factor=True factor.kwargs.weight=10.0 -m student2.cp.jku.at:27017:better_priors 
# Orthogonal
python -m experiments.vae_experiment with -p prior_class=priors.OrthogonalPrior prior.kwargs.weight=1000.0 -m student2.cp.jku.at:27017:better_priors 
# Simplex
python -m experiments.vae_experiment with -p prior_class=priors.SimplexPrior prior.kwargs.weight=1000.0 -m student2.cp.jku.at:27017:better_priors 
# BETA TCVAE
python -m experiments.vae_experiment with -p prior_class=priors.BetaTCVaePrior prior.kwargs.weight=1.0 -m student2.cp.jku.at:27017:better_priors 

```

4. Results are in MongoDB and experiment_logs folder


## TODOs
- Testing
- Typing
- Datasets
- Priors
- Models
- Make model more readable
- ...