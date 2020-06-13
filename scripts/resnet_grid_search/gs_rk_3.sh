epochs=100
loss_class=BCE
valid_types=loose

for learning_rate in 1e-4 1e-5
do
  for rf in a_bit_larger normal a_bit_smaller
  do
    ./scripts/per_id_run_parallel.sh classification_experiment "id=resnet_gridsearch_${rf}_${valid_types}_${learning_rate}_${epochs}_${loss_class}  learning_rate=$learning_rate epochs=$epochs rf=$rf valid_types=$valid_types loss_class=dcase2020_task2.losses.$loss_class -m student2.cp.jku.at:27017:resnet_gridsearch"
  done
done
