epochs=100
machine_type=3
machine_id=1

for rf in a_bit_larger
do
  for valid_types in loose
  do
    for learning_rate_decay in 0.98 0.97 0.96
      python -m dcase2020_task2.experiments.classification_experiment with learning_rate_decay=$learning_rate_decay machine_type=$machine_type machine_id=$machine_id epochs=$epochs rf=$rf valid_types=$valid_types -m student2.cp.jku.at:27017:debug_
    done
  done
done
