conda activate dcase2020_task2

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=0 machine_id=0 $2 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=0 machine_id=2 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=0 machine_id=4 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=0 machine_id=6 $2 > /dev/null 2>&1 &

wait

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=1 machine_id=0 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=1 machine_id=2 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=1 machine_id=4 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=1 machine_id=6 $2 > /dev/null 2>&1 &

wait

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=2 machine_id=0 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=2 machine_id=2 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=2 machine_id=4 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=2 machine_id=6 $2 > /dev/null 2>&1 &

wait

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=3 machine_id=1 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=3 machine_id=2 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=3 machine_id=3 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=3 machine_id=4 $2 > /dev/null 2>&1 &

wait

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=4 machine_id=1 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=4 machine_id=2 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=4 machine_id=3 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=5 machine_id=0 $2 > /dev/null 2>&1 &

wait

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=5 machine_id=2 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=5 machine_id=4 $2 > /dev/null 2>&1 &
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python -m dcase2020_task2.experiments.$1 with num_workers=4 machine_type=5 machine_id=6 $2 > /dev/null 2>&1 &
