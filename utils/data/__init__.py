
def get_dataset(dataset_name):
    if dataset_name == 'genjacquard':
        from .GenJacquard_data import GenJacquardDataset
        return GenJacquardDataset

    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))