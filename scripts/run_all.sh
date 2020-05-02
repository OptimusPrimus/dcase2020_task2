cd ..
cd dcase2020_task2
conda activate dcase2020_task2


OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=0 machine_id=0  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=0 machine_id=2  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=0 machine_id=4  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=0 machine_id=6  -m student2.cp.jku.at:27017:dcase2020_2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=1 machine_id=0  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=1 machine_id=2  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=1 machine_id=4  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=1 machine_id=6  -m student2.cp.jku.at:27017:dcase2020_2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=2 machine_id=0  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=2 machine_id=2  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=2 machine_id=4  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=2 machine_id=6  -m student2.cp.jku.at:27017:dcase2020_2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=3 machine_id=1  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=3 machine_id=2  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=3 machine_id=3  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=3 machine_id=4  -m student2.cp.jku.at:27017:dcase2020_2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=4 machine_id=1  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=4 machine_id=2  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=4 machine_id=3  -m student2.cp.jku.at:27017:dcase2020_2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=5 machine_id=0  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=5 machine_id=2  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=5 machine_id=4  -m student2.cp.jku.at:27017:dcase2020_2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$2 python -m experiments.$1 with num_workers=4 machine_type=5 machine_id=6  -m student2.cp.jku.at:27017:dcase2020_2