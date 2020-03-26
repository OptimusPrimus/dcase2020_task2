cd ..
cd dcase2020_task2
conda activate dcase2020_task2

for MACHINE_TYPE in 0 1 2 3 4 5
do
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.simple_sampling_experiment with latent_size=8 learning_rate=0.0001 weight_decay=0.0001 normalize=True feature_context=long reconstruction_class=reconstructions.NP model_class=models.SamplingFCAE machine_type=$MACHINE_TYPE num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2_simple_sampling > /dev/null 2>&1 &
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.simple_sampling_experiment with latent_size=40 learning_rate=0.0001 weight_decay=0.0001 normalize=True feature_context=long reconstruction_class=reconstructions.NP model_class=models.SamplingFCAE machine_type=$MACHINE_TYPE num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2_simple_sampling > /dev/null 2>&1 &
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.simple_sampling_experiment with latent_size=8 learning_rate=0.0001 weight_decay=0.0001 normalize=False feature_context=long reconstruction_class=reconstructions.NP model_class=models.SamplingFCAE machine_type=$MACHINE_TYPE num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2_simple_sampling > /dev/null 2>&1 &
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.simple_sampling_experiment with latent_size=8 learning_rate=0.0001 weight_decay=0.0001 normalize=True rho=0.1 feature_context=long reconstruction_class=reconstructions.NP model_class=models.SamplingFCAE machine_type=$MACHINE_TYPE num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2_simple_sampling > /dev/null 2>&1 &

  wait

  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.simple_sampling_experiment with latent_size=8 learning_rate=0.0001 weight_decay=0.0001 normalize=True feature_context=short reconstruction_class=reconstructions.NP model_class=models.SamplingFCAE machine_type=$MACHINE_TYPE num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2_simple_sampling > /dev/null 2>&1 &
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.simple_sampling_experiment with latent_size=8 learning_rate=0.0001 weight_decay=0.0001 normalize=True feature_context=long reconstruction_class=reconstructions.MSE model_class=models.SamplingFCAE machine_type=$MACHINE_TYPE num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2_simple_sampling > /dev/null 2>&1 &
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.simple_sampling_experiment with latent_size=8 learning_rate=0.0001 weight_decay=0.0001 normalize=True feature_context=long reconstruction_class=reconstructions.NP model_class=models.BaselineFCAE machine_type=$MACHINE_TYPE num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2_simple_sampling > /dev/null 2>&1 &
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.simple_sampling_experiment with latent_size=8 learning_rate=0.0001 weight_decay=0.0 normalize=True feature_context=long reconstruction_class=reconstructions.NP model_class=models.SamplingFCAE machine_type=$MACHINE_TYPE num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2_simple_sampling > /dev/null 2>&1 &

  wait
done

wait