conda activate dcase2020_task2

./scripts/run_all.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=256 batch_size=512 learning_rate=1e-4 weight_decay=0 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=256 batch_size=512 learning_rate=1e-5 weight_decay=0 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=256 batch_size=512 learning_rate=1e-4 weight_decay=1e-5 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=256 batch_size=512 learning_rate=1e-5 weight_decay=1e-5 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &

wait

./scripts/run_all.sh classification_experiment 0 "debug=False num_hidden=3 hidden_size=512 batch_size=512 learning_rate=1e-4 weight_decay=0 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 1 "debug=False num_hidden=3 hidden_size=512 batch_size=512 learning_rate=1e-5 weight_decay=0 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 2 "debug=False num_hidden=3 hidden_size=512 batch_size=512 learning_rate=1e-4 weight_decay=1e-5 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 3 "debug=False num_hidden=3 hidden_size=512 batch_size=512 learning_rate=1e-5 weight_decay=1e-5 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &

wait

./scripts/run_all.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=256 batch_size=512 learning_rate=1e-4 weight_decay=0 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=256 batch_size=512 learning_rate=1e-5 weight_decay=0 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=256 batch_size=512 learning_rate=1e-4 weight_decay=1e-5 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=256 batch_size=512 learning_rate=1e-5 weight_decay=1e-5 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &

wait

./scripts/run_all.sh classification_experiment 0 "debug=False num_hidden=4 hidden_size=512 batch_size=512 learning_rate=1e-4 weight_decay=0 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 1 "debug=False num_hidden=4 hidden_size=512 batch_size=512 learning_rate=1e-5 weight_decay=0 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 2 "debug=False num_hidden=4 hidden_size=512 batch_size=512 learning_rate=1e-4 weight_decay=1e-5 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &
./scripts/run_all.sh classification_experiment 3 "debug=False num_hidden=4 hidden_size=512 batch_size=512 learning_rate=1e-5 weight_decay=1e-5 model_class=dcase2020_task2.models.CNN loss_class=dcase2020_task2.losses.AUC -m student2.cp.jku.at:27017:dcase2020_task2_conv_complement_classification_gridsearch" &

wait