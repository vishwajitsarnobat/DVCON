import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import TaskAwareYOLO
from ultralytics.utils.ops import non_max_suppression, scale_boxes

# 1. Force CPU execution as required by Stage 2A guidelines
device = torch.device('cpu')
print(f"Running inference on: {device}")

# 2. Load your custom DENSE model
print("Loading Trained Dense TaskAwareYOLO...")
model = TaskAwareYOLO(embedding_path='task_embeddings.pt').to(device)
model.load_state_dict(torch.load('yolov8n_dense_trained.pt', map_location=device))
model.eval()

def run_inference(image_path, query_text, task_id):
    print(f"\nProcessing Query: '{query_text}' (Task ID: {task_id})")
    
    # Prep the image (YOLO expects 1x3x640x640 normalized tensor)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    # Prep the task ID
    task_tensor = torch.tensor([task_id], dtype=torch.long).to(device)
    
    # Run through your Fusion Engine
    with torch.no_grad():
        predictions = model(img_tensor, task_tensor)
        
        # ROBUSTNESS FIX: NMS Compatibility
        # If YOLO's Detect head returns a tuple during eval mode, we extract just the inference tensor
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
            
        # Apply NMS (Non-Max Suppression) to filter overlapping boxes (Class 0 is preferred object)
        boxes = non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45, classes=[0])[0]

    # Plotting
    plt.figure(figsize=(10, 8))
    
    if len(boxes):
        # Scale boxes back to original image size
        boxes[:, :4] = scale_boxes((640, 640), boxes[:, :4], img_rgb.shape).round()
        
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw Box
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"Match {conf:.2f}"
            cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        print("No matches found above confidence threshold.")

    plt.imshow(img_rgb)
    plt.title(f"Query: {query_text}")
    plt.axis('off')
    
    output_filename = f"output_task_{task_id}.jpg"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Saved snapshot to {output_filename}")
    plt.show()

if __name__ == "__main__":
    # Ensure you replace these with your actual test images from the 14 queries!
    test_cases = [
        {"text": "What should I use to cut a paper?", "task_id": 0, "img": "test_image_1.jpg"},
        {"text": "Where can I park my vehicle?", "task_id": 1, "img": "test_image_2.jpg"},
    ]
    
    for case in test_cases:
        try:
            run_inference(case["img"], case["text"], case["task_id"])
        except Exception as e:
            print(f"Failed on {case['img']}: {e}")