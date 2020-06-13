epochs=150

for rf in a_bit_larger normal a_bit_smaller
do
  for valid_types in loose
  do
    python -m dcase2020_task2.experiments.classification_experiment with epochs=$epochs rf=$rf valid_types=$valid_types -m student2.cp.jku.at:27017:debug_
  done
done
