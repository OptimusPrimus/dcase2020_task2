cd ..
cd dcase2020_task2/dcase2020_task2
conda activate dcase2020_task2


OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with machine_type=0 latent_size=32 -m student2.cp.jku.at:27017:dcase2020_task2&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with machine_type=1 latent_size=32 -m student2.cp.jku.at:27017:dcase2020_task2&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with machine_type=2 latent_size=32 -m student2.cp.jku.at:27017:dcase2020_task2&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with machine_type=3 latent_size=32 -m student2.cp.jku.at:27017:dcase2020_task2&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.baseline_experiment with machine_type=4 latent_size=32 -m student2.cp.jku.at:27017:dcase2020_task2&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.baseline_experiment with machine_type=5 latent_size=32 -m student2.cp.jku.at:27017:dcase2020_task2&

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.baseline_experiment with machine_type=0 latent_size=32 batch_size=256 -m student2.cp.jku.at:27017:dcase2020_task2&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.baseline_experiment with machine_type=1 latent_size=32 batch_size=256 -m student2.cp.jku.at:27017:dcase2020_task2&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m experiments.baseline_experiment with machine_type=2 latent_size=32 batch_size=256 -m student2.cp.jku.at:27017:dcase2020_task2&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m experiments.baseline_experiment with machine_type=3 latent_size=32 batch_size=256 -m student2.cp.jku.at:27017:dcase2020_task2&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m experiments.baseline_experiment with machine_type=4 latent_size=32 batch_size=256 -m student2.cp.jku.at:27017:dcase2020_task2&
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m experiments.baseline_experiment with machine_type=5 latent_size=32 batch_size=256 -m student2.cp.jku.at:27017:dcase2020_task2&

wait