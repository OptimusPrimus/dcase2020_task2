epochs=100
machine_type=3
machine_id=1

for rf in a_bit_larger
do
  for valid_types in loose
  do
    for learning_rate_decay in 0.97
    do
      for hidden_size in 128 256
      do
        python -m dcase2020_task2.experiments.classification_experiment with learning_rate_decay=$learning_rate_decay machine_type=0 machine_id=0 epochs=$epochs rf=$rf valid_types=$valid_types hidden_size=$hidden_size -m student2.cp.jku.at:27017:debug_
        python -m dcase2020_task2.experiments.classification_experiment with learning_rate_decay=$learning_rate_decay machine_type=1 machine_id=0 epochs=$epochs rf=$rf valid_types=$valid_types hidden_size=$hidden_size -m student2.cp.jku.at:27017:debug_
        python -m dcase2020_task2.experiments.classification_experiment with learning_rate_decay=$learning_rate_decay machine_type=2 machine_id=0 epochs=$epochs rf=$rf valid_types=$valid_types hidden_size=$hidden_size -m student2.cp.jku.at:27017:debug_
        python -m dcase2020_task2.experiments.classification_experiment with learning_rate_decay=$learning_rate_decay machine_type=3 machine_id=1 epochs=$epochs rf=$rf valid_types=$valid_types hidden_size=$hidden_size -m student2.cp.jku.at:27017:debug_
        python -m dcase2020_task2.experiments.classification_experiment with learning_rate_decay=$learning_rate_decay machine_type=4 machine_id=1 epochs=$epochs rf=$rf valid_types=$valid_types hidden_size=$hidden_size -m student2.cp.jku.at:27017:debug_
        python -m dcase2020_task2.experiments.classification_experiment with learning_rate_decay=$learning_rate_decay machine_type=5 machine_id=0 epochs=$epochs rf=$rf valid_types=$valid_types hidden_size=$hidden_size -m student2.cp.jku.at:27017:debug_
      done
    done
  done
done
