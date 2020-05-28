conda activate dcase2020_task2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 python -m dcase2020_task2.experiments.baseline_dcase_experiment with num_workers=4 machine_type=0 -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 python -m dcase2020_task2.experiments.baseline_dcase_experiment with num_workers=4 machine_type=1 -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 python -m dcase2020_task2.experiments.baseline_dcase_experiment with num_workers=4 machine_type=2 -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 python -m dcase2020_task2.experiments.baseline_dcase_experiment with num_workers=4 machine_type=3 -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 python -m dcase2020_task2.experiments.baseline_dcase_experiment with num_workers=4 machine_type=4 -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 python -m dcase2020_task2.experiments.baseline_dcase_experiment with num_workers=4 machine_type=5 -m student2.cp.jku.at:27017:dcase2020_2
