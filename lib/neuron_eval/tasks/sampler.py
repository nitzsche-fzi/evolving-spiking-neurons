import torch

class MultiEpochBatchSampler(torch.utils.data.sampler.Sampler):
    """
    A sampler that allows for multiple epochs over the dataset.
    This is useful for tasks where you want to iterate over the dataset multiple times.
    """
    def __init__(self, data_source, batch_size, num_epochs):
        super(MultiEpochBatchSampler, self).__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_samples = len(data_source)

    def __iter__(self):
        to_yield = []
        for _ in range(self.num_epochs):
            indices = torch.randperm(self.num_samples).tolist()
            for i in indices:
                to_yield.append(i)
                if len(to_yield) == self.batch_size:
                    yield to_yield
                    to_yield = []

    def __len__(self):
        return (self.num_samples * self.num_epochs) // self.batch_size

