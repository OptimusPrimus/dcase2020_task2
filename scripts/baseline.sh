cd ..
cd dcase2020_task2
conda activate dcase2020_task2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with num_workers=2 machine_type=0 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with num_workers=2 machine_type=1 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with num_workers=2 machine_type=2 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.baseline_experiment with num_workers=2 machine_type=3 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.baseline_experiment with num_workers=2 machine_type=4 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m experiments.baseline_experiment with num_workers=2 machine_type=5 -m student2.cp.jku.at:27017:dcase2020_task2 > /dev/null 2>&1 &

wait
