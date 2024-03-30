from data.TUs import TUsDataset


def LoadData(DATASET_NAME):

    TU_DATASETS = ['DD', 'ENZYMES']
    if DATASET_NAME in TU_DATASETS: 
        return TUsDataset(DATASET_NAME)
