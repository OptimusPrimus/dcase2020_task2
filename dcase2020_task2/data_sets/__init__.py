
CLASS_MAP = {
    'fan': 0,
    'pump': 1,
    'slider': 2,
    'ToyCar': 3,
    'ToyConveyor': 4,
    'valve': 5
}
INVERSE_CLASS_MAP = {
    0: 'fan',
    1: 'pump',
    2: 'slider',
    3: 'ToyCar',
    4: 'ToyConveyor',
    5: 'valve'
}
TRAINING_ID_MAP = {
    0: [0, 2, 4, 6],
    1: [0, 2, 4, 6],
    2: [0, 2, 4, 6],
    3: [1, 2, 3, 4],
    4: [1, 2, 3],
    5: [0, 2, 4, 6]
}
EVALUATION_ID_MAP = {
    0: [1, 3, 5],
    1: [1, 3, 5],
    2: [1, 3, 5],
    3: [5, 6, 7],
    4: [4, 5, 6],
    5: [1, 3, 5]
}
ALL_ID_MAP = {
    0: [0, 1, 2, 3, 4, 5, 6],
    1: [0, 1, 2, 3, 4, 5, 6],
    2: [0, 1, 2, 3, 4, 5, 6],
    3: [1, 2, 3, 4, 5, 6, 7],
    4: [1, 2, 3, 4, 5, 6],
    5: [0, 1, 2, 3, 4, 5, 6]
}


def enumerate_development_datasets():
    typ_id = []
    for i in range(6):
        for j in TRAINING_ID_MAP[i]:
            typ_id.append((i, j))
    return typ_id


def enumerate_evaluation_datasets():
    typ_id = []
    for i in range(6):
        for j in EVALUATION_ID_MAP[i]:
            typ_id.append((i, j))
    return typ_id


from dcase2020_task2.data_sets.base_data_set import BaseDataSet
from dcase2020_task2.data_sets.mcm_dataset import MCMDataSet, MachineDataSet
from dcase2020_task2.data_sets.complement_dataset import ComplementMCMDataSet
