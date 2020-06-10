conda activate dcase2020_task2

./scripts/per_id_create_submission.sh classification_experiment "id=cnn_classification_machine_data_set debug=False same_type=True num_hidden=3 hidden_size=256 batch_size=512 learning_rate=1e-4 weight_decay=0 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_submission"