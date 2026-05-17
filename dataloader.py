import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class COCOTasksDataset(Dataset):
    def __init__(self, annotation_dir, image_dir, img_size=640):
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.img_size = img_size
        
        # Transform to resize image and convert to PyTorch Tensor (C, H, W) normalized 0-1
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        # Parse all 14 task JSONs
        print("Parsing COCO-Tasks annotations...")
        self.data = self._load_annotations()
        print(f"Loaded {len(self.data)} valid training samples.")

    def _load_annotations(self):
        dataset_items = []
        
        # Loop through all 14 tasks
        for task_id in range(1, 15): # Tasks 1 to 14
            json_file = os.path.join(self.annotation_dir, f"task_{task_id}_train.json")
            if not os.path.exists(json_file):
                print(f"Failed to open file at index number {task_id}")
                continue
                
            with open(json_file, 'r') as f:
                task_data = json.load(f)
                
            # Create a dictionary to group annotations by image_id
            images_dict = {img['id']: img for img in task_data.get('images', [])}
            
            # Loop through annotations to find "preferred" objects
            for ann in task_data.get('annotations', []):
                # category_id: 1 means preferred object for this task!
                if ann['category_id'] == 1:
                    img_info = images_dict.get(ann['image_id'])
                    if not img_info:
                        continue
                        
                    # Build the data packet
                    dataset_items.append({
                        'img_name': img_info['file_name'],
                        'orig_w': img_info['width'],
                        'orig_h': img_info['height'],
                        'bbox': ann['bbox'], # [x_min, y_min, width, height]
                        'task_id': task_id - 1 # Zero-index the task (0 to 13) for the Embedding layer
                    })
                    
        return dataset_items
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Load Image
        img_path = os.path.join(self.image_dir, item['img_name'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # 2. Convert BBox to YOLO format (Normalized x_center, y_center, width, height)
        x_min, y_min, w, h = item['bbox']
        orig_w, orig_h = item['orig_w'], item['orig_h']
        
        x_center = (x_min + w / 2.0) / orig_w
        y_center = (y_min + h / 2.0) / orig_h
        norm_w = w / orig_w
        norm_h = h / orig_h
        
        # YOLO target format: [class_id, x_center, y_center, w, h]
        # Since we only detect ONE "preferred" class per task, class_id is always 0
        target = torch.tensor([0.0, x_center, y_center, norm_w, norm_h], dtype=torch.float32)
        
        # 3. Load Task ID
        task_id_tensor = torch.tensor(item['task_id'], dtype=torch.long)
        
        return image, target, task_id_tensor

# Collate function needed to batch varying number of targets (even if it's 1 per image)
def collate_fn(batch):
    images, targets, task_ids = zip(*batch)
    
    # YOLO loss expects targets in shape (num_boxes, 6) -> [batch_idx, class, x, y, w, h]
    batched_targets = []
    for i, target in enumerate(targets):
        # Prepend the batch index 'i' to the target row
        target_with_batch = torch.cat([torch.tensor([i], dtype=torch.float32), target])
        batched_targets.append(target_with_batch)
        
    return torch.stack(images), torch.stack(batched_targets), torch.stack(task_ids)