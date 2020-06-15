epochs=100
loss_class=BCE
valid_types=loose
hidden_size=128

for learning_rate in 1e-4
do
  for rf in a_bit_larger
  do
    for learning_rate_decay in 0.99
    do
        ./scripts/per_id_run_parallel.sh classification_experiment "id=resnet_gridsearch_2_${rf}_${valid_types}_${learning_rate}_${learning_rate_decay}_${epochs}_${loss_class}  hidden_size=$hidden_size learning_rate=$learning_rate learning_rate_decay=$learning_rate_decay epochs=$epochs rf=$rf valid_types=$valid_types loss_class=dcase2020_task2.losses.$loss_class -m student2.cp.jku.at:27017:resnet_gridsearch"
    done
  done
done
