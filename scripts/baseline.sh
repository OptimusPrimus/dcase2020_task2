cd ..
cd dcase2020_task2
conda activate dcase2020_task2

# with normalization
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with num_workers=2 normalize=True machine_type=0 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with num_workers=2 normalize=True machine_type=1 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with num_workers=2 normalize=True machine_type=2 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with num_workers=2 normalize=True machine_type=3 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.baseline_experiment with num_workers=2 normalize=True machine_type=4 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.baseline_experiment with num_workers=2 normalize=True machine_type=5 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &

# without normalization
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.baseline_experiment with num_workers=2 normalize=False machine_type=0 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.baseline_experiment with num_workers=2 normalize=False machine_type=1 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m experiments.baseline_experiment with num_workers=2 normalize=False machine_type=2 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m experiments.baseline_experiment with num_workers=2 normalize=False machine_type=3 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m experiments.baseline_experiment with num_workers=2 normalize=False machine_type=4 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m experiments.baseline_experiment with num_workers=2 normalize=False machine_type=5 latent_size=8 learning_rate=0.001 -m student2.cp.jku.at:27017:dcase2020_task2_baseline > /dev/null 2>&1 &


wait