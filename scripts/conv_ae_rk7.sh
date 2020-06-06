conda activate dcase2020_task2

#./scripts/run_all.sh baseline_experiment 0 "debug=False num_hidden=1 hidden_size=128 latent_size=4 weight_decay=0 model_class=dcase2020_task2.models.ConvAE reconstruction_class=dcase2020_task2.losses.MSEReconstruction -m student2.cp.jku.at:27017:dcase2020_task2_ae_baseline_gridsearch" &
#./scripts/run_all.sh baseline_experiment 1 "debug=False num_hidden=1 hidden_size=128 latent_size=4 weight_decay=1e-5 model_class=dcase2020_task2.models.ConvAE reconstruction_class=dcase2020_task2.losses.MSEReconstruction -m student2.cp.jku.at:27017:dcase2020_task2_ae_baseline_gridsearch" &
#./scripts/run_all.sh baseline_experiment 2 "debug=False num_hidden=1 hidden_size=128 latent_size=8 weight_decay=0 model_class=dcase2020_task2.models.ConvAE reconstruction_class=dcase2020_task2.losses.MSEReconstruction -m student2.cp.jku.at:27017:dcase2020_task2_ae_baseline_gridsearch" &
#./scripts/run_all.sh baseline_experiment 3 "debug=False num_hidden=1 hidden_size=128 latent_size=8 weight_decay=1e-5 model_class=dcase2020_task2.models.ConvAE reconstruction_class=dcase2020_task2.losses.MSEReconstruction -m student2.cp.jku.at:27017:dcase2020_task2_ae_baseline_gridsearch" &

./scripts/run_all.sh baseline_experiment 0 "debug=False num_hidden=1 hidden_size=256 latent_size=4 weight_decay=0 model_class=dcase2020_task2.models.ConvAE reconstruction_class=dcase2020_task2.losses.MSEReconstruction -m student2.cp.jku.at:27017:dcase2020_task2_ae_baseline_gridsearch" &
./scripts/run_all.sh baseline_experiment 1 "debug=False num_hidden=1 hidden_size=256 latent_size=4 weight_decay=1e-5 model_class=dcase2020_task2.models.ConvAE reconstruction_class=dcase2020_task2.losses.MSEReconstruction -m student2.cp.jku.at:27017:dcase2020_task2_ae_baseline_gridsearch" &
./scripts/run_all.sh baseline_experiment 2 "debug=False num_hidden=1 hidden_size=256 latent_size=8 weight_decay=0 model_class=dcase2020_task2.models.ConvAE reconstruction_class=dcase2020_task2.losses.MSEReconstruction -m student2.cp.jku.at:27017:dcase2020_task2_ae_baseline_gridsearch" &
./scripts/run_all.sh baseline_experiment 3 "debug=False num_hidden=1 hidden_size=256 latent_size=8 weight_decay=1e-5 model_class=dcase2020_task2.models.ConvAE reconstruction_class=dcase2020_task2.losses.MSEReconstruction -m student2.cp.jku.at:27017:dcase2020_task2_ae_baseline_gridsearch" &

