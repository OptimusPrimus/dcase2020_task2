conda activate dcase2020_task2

# batch sizes [4096, 8192]
# weight decay [ 0 ]

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=128 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=128 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=128 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=128 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=256 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=256 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=256 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=256 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=512 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=512 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=512 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=512 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=128 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=128 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=128 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=128 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=256 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=256 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=256 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=256 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=512 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=512 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=512 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=512 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

# weight decay [1e-5]

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=128 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=128 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=128 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=128 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=256 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=256 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=256 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=256 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=512 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=512 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=512 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=512 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=128 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=128 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=128 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=128 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=256 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=256 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=256 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=256 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=512 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=512 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=512 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=512 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

# batch sizes [1024, 2048]
# weight decay [ 0 ]

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=128 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=128 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=128 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=128 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=256 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=256 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=256 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=256 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=512 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=512 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=512 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=512 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=128 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=128 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=128 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=128 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=256 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=256 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=256 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=256 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=512 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=512 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=512 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=512 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

# weight decay [1e-5]

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=128 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=128 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=128 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=128 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=256 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=256 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=256 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=256 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=512 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=512 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=512 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=512 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=128 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=128 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=128 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=128 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=256 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=256 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=256 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=256 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=512 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=512 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=512 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=512 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

# learning rate


# batch sizes [4096, 8192]
# weight decay [ 0 ]

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=4096 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=8192 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

# weight decay [1e-5]

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=8192 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

# batch sizes [1024, 2048]
# weight decay [ 0 ]

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=1024 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=2048 weight_decay=0 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

# weight decay [1e-5]

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=128 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=256 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=3 hidden_size=512 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=128 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=256 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

./scripts/per_id_run.sh classification_experiment 0 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 1 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=1024 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 2 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &
./scripts/per_id_run.sh classification_experiment 3 "debug=False learning_rate=1e-3 num_hidden=4 hidden_size=512 batch_size=2048 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.BCE -m student2.cp.jku.at:27017:dcase2020_task2_complement_classification_gridsearch" &

wait