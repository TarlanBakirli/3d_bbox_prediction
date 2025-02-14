from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from data.dataset import BBoxPredictionDataset
from data.transforms import ToTensor

def get_dataloader(split, test_size=0.2):

    bboxpredictiondataset = BBoxPredictionDataset("../dl_challenge",
                      transforms.Compose([ToTensor()]))

    split_thrsh = int(bboxpredictiondataset.__len__() * test_size)
    indices = range(bboxpredictiondataset.__len__())

    train_indices, test_indices = indices[split_thrsh:], indices[:split_thrsh]

    if split == "train":
        sampler = SubsetRandomSampler(train_indices)
    else:
        sampler = SubsetRandomSampler(test_indices)
    
    dataloader = DataLoader(bboxpredictiondataset, batch_size=1, sampler=sampler, num_workers=0)
    
    return dataloader