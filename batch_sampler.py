import torch
from mydataloader.sampler import Sampler
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self):
        self.img = torch.cat([torch.ones(2, 2) for i in range(100)], dim=0)
        self.num_classes = 2
        self.label = torch.tensor(
            [random.randint(0, self.num_classes - 1) for i in range(100)]
        )

    def __getitem__(self, index):
        return self.img[index], self.label[index]

    def __len__(self):
        return len(self.label)

class CustomSampler(Sampler):
    def __init__(self, sampler, data,batch_size):
        self.sampler = sampler
        self.data = data
        self.len=len(data)
        self.bs=batch_size

    def __iter__(self):
        for i in range(self.len-self.bs+1):
            batch=self.data[i:i+self.bs-1]
            yield batch


    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    data=Data()
    batch_s=CustomSampler(data,4)
    dataloader = DataLoader(data, num_workers = 4, pin_memory=True,\
                                        batch_sampler=batch_s)
    print(1)