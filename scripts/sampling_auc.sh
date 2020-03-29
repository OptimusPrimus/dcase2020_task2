cd ..
cd dcase2020_task2
conda activate dcase2020_task2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.sampling_experiment with mse_weight=0.0 machine_type=0 num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.sampling_experiment with mse_weight=0.0 machine_type=1 num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.sampling_experiment with mse_weight=0.0 machine_type=2 num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.sampling_experiment with mse_weight=0.0 machine_type=3 num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.sampling_experiment with mse_weight=0.0 machine_type=4 num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m experiments.sampling_experiment with mse_weight=0.0 machine_type=5 num_workers=2 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &

wait