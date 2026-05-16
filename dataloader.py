import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transformers as transforms

class COCOTasksDataset(Dataset):
    def __init__(self,annotation_file,image_dir,transform=None):
        """

        """
        self.data=self._load_annotations(annotation_file)
        self.image_dir=self.image_dir
        self.transform=transform

    def _load_annotations(self,file_path):
        # [{'img': '001.jpg', 'targets': [...], 'task_id': 3}, ...]
        return []
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        item=self.data[idx]