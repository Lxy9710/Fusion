import torch
from mydataloader.self_dataloader import DataLoader

class MySampler(Sampler):
    def __init__(self, data_source, indices):
        super(MySampler, self).__init__(data_source)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


for cls in range(5):
    indices, = torch.where(labels == cls)
    my_sampler = MySampler(dataset, indices)
    loader = DataLoader(dataset,
                        batch_size=10,
                        sampler=my_sampler,
                        drop_last=True)

    for batch_data, batch_labels in loader:
        # Do you training / evaluating here
        pass
