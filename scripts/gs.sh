#!/usr/bin/zsh

cd ..
cd dcase2020_task2
conda activate dcase2020_task2

#
for MACHINE_TYPE in 0 1 2 3 4 5
do
  for LEARNING_RATE in 0.001 0.0001
  do
    for FEATURE_CONTEXT in "short" "long"
    do
      for LATENT_SIZE in 8 40
      do
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.simple_sampling_experiment with machine_type=$MACHINE_TYPE latent_size=$LATENT_SIZE reconstruction_class=reconstructions.NP model_class=models.BaselineFCAE feature_context=$FEATURE_CONTEXT learning_rate=$LEARNING_RATE -m student2.cp.jku.at:27017:test > /dev/null 2>&1 &
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.simple_sampling_experiment with machine_type=$MACHINE_TYPE latent_size=$LATENT_SIZE reconstruction_class=reconstructions.MSE model_class=models.BaselineFCAE feature_context=$FEATURE_CONTEXT learning_rate=$LEARNING_RATE -m student2.cp.jku.at:27017:test > /dev/null 2>&1 &
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m experiments.simple_sampling_experiment with machine_type=$MACHINE_TYPE latent_size=$LATENT_SIZE reconstruction_class=reconstructions.NP model_class=models.SamplingFCAE feature_context=$FEATURE_CONTEXT learning_rate=$LEARNING_RATE -m student2.cp.jku.at:27017:test > /dev/null 2>&1 &
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.simple_sampling_experiment with machine_type=$MACHINE_TYPE latent_size=$LATENT_SIZE reconstruction_class=reconstructions.MSE model_class=models.SamplingFCAE feature_context=$FEATURE_CONTEXT learning_rate=$LEARNING_RATE -m student2.cp.jku.at:27017:test > /dev/null 2>&1 &
      done
    done
    wait
  done
done