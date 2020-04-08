cd ..
cd dcase2020_task2
conda activate dcase2020_task2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=0 machine_id=0 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=0 machine_id=2 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=0 machine_id=4 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=0 machine_id=6 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=1 machine_id=0 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=1 machine_id=2 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=1 machine_id=4 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=1 machine_id=6 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=2 machine_id=0 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=2 machine_id=2 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=2 machine_id=4 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=2 machine_id=6 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=3 machine_id=1 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=3 machine_id=2 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=3 machine_id=3 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=3 machine_id=4 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=4 machine_id=1 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=4 machine_id=2 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=4 machine_id=3 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=5 machine_id=0 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=5 machine_id=2 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=5 machine_id=4 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$3 python -m experiments.classification_experiment with num_workers=4 machine_type=5 machine_id=6 normalize=$1 complement=$2 descriptor=$4 -m student2.cp.jku.at:27017:dcase2020_task2

