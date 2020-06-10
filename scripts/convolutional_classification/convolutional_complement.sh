conda activate dcase2020_task2

./scripts/per_id_run_parallel.sh classification_experiment "debug=False num_hidden=3 hidden_size=256 batch_size=512 learning_rate=1e-4 weight_decay=0 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch"