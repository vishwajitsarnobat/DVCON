import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class COCOTasksDataset(Dataset):
    def __init__(self, annotation_dir, image_dir, img_size=640):
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.img_size = img_size

        # Letterbox transform (resize with padding) instead of direct resize
        self.transform = transforms.Compose([
            transforms.ToTensor()   # will be applied after letterbox
        ])

        print("Parsing COCO-Tasks annotations...")
        self.data = self._load_annotations()
        print(f"Loaded {len(self.data)} valid training samples.")

    def _load_annotations(self):
        dataset_items = []
        for task_id in range(1, 15):
            json_file = os.path.join(self.annotation_dir, f"task_{task_id}_train.json")
            if not os.path.exists(json_file):
                print(f"Failed to open file at index number {task_id}")
                continue
            with open(json_file, 'r') as f:
                task_data = json.load(f)
            images_dict = {img['id']: img for img in task_data.get('images', [])}
            for ann in task_data.get('annotations', []):
                if ann['category_id'] == 1:
                    img_info = images_dict.get(ann['image_id'])
                    if not img_info:
                        continue
                    dataset_items.append({
                        'img_name': img_info['file_name'],
                        'orig_w': img_info['width'],
                        'orig_h': img_info['height'],
                        'bbox': ann['bbox'],
                        'task_id': task_id - 1
                    })
        return dataset_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        img_path = os.path.join(self.image_dir, item['img_name'])
        image = Image.open(img_path).convert("RGB")

        # --- NEW: Letterbox resize to self.img_size (keeping aspect ratio, pad with grey) ---
        original_w, original_h = image.size
        scale = min(self.img_size / original_w, self.img_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        image = transforms.Resize((new_h, new_w))(image)
        # Pad to square
        pad_left = (self.img_size - new_w) // 2
        pad_top = (self.img_size - new_h) // 2
        pad_right = self.img_size - new_w - pad_left
        pad_bottom = self.img_size - new_h - pad_top
        image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=128)  # grey padding
        image = transforms.ToTensor()(image)   # now shape (3,640,640)

        # Convert bbox to YOLO format (normalized, with respect to original image dimensions)
        x_min, y_min, w, h = item['bbox']
        orig_w, orig_h = item['orig_w'], item['orig_h']
        x_center = (x_min + w / 2.0) / orig_w
        y_center = (y_min + h / 2.0) / orig_h
        norm_w = w / orig_w
        norm_h = h / orig_h

        target = torch.tensor([0.0, x_center, y_center, norm_w, norm_h], dtype=torch.float32)
        task_id_tensor = torch.tensor(item['task_id'], dtype=torch.long)

        return image, target, task_id_tensor

def collate_fn(batch):
    images, targets, task_ids = zip(*batch)
    batched_targets = []
    for i, target in enumerate(targets):
        target_with_batch = torch.cat([torch.tensor([i], dtype=torch.float32), target])
        batched_targets.append(target_with_batch)
    return torch.stack(images), torch.stack(batched_targets), torch.stack(task_ids)