conda activate dcase2020_task2

./scripts/create_submission.sh classification_experiment "id=flat_classification_machine_data_set same_type=True debug=False num_hidden=3 hidden_size=128 batch_size=4096 weight_decay=1e-5 model_class=dcase2020_task2.models.FCNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_submission"