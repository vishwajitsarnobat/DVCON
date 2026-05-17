import torch
import torch.nn as nn
from ultralytics import YOLO
from prune_utils import apply_2_4_sparsity

class TaskGuidedFusionEngine(nn.Module):
    def __init__(self, feature_channels, embedding_dim=384):
        super().__init__()
        self.channel_projection = nn.Linear(embedding_dim, feature_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, task_embedding):
        # x: (B,C,H,W), task_embedding: (B,384)
        projected_task = self.channel_projection(task_embedding)
        projected_task = projected_task.unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        mask = self.sigmoid(projected_task)
        if not self.training:
            mask = (mask >= 0.5).float()
        return x * mask

class TaskAwareYOLO(nn.Module):
    def __init__(self, base_model_name='yolov8n.pt', embedding_path='task_embeddings.pt'):
        super().__init__()
        yolo_wrapper = YOLO('yolov8n.pt')
        self.core_model = yolo_wrapper.model   # original DetectionModel
        self.task_embeddings = nn.Embedding.from_pretrained(
            torch.load(embedding_path, map_location='cpu'), freeze=True
        )  # shape (14,384)

        # Create fusion engines for the three feature levels
        self.fusion_p3 = TaskGuidedFusionEngine(64, 384)
        self.fusion_p4 = TaskGuidedFusionEngine(128, 384)
        self.fusion_p5 = TaskGuidedFusionEngine(256, 384)

        # Register forward hooks on the layers after which we want to apply gating
        self._register_fusion_hooks()

    def _register_fusion_hooks(self):
        # Indices of layers that produce P3, P4, P5 feature maps in YOLOv8n
        # These indices are obtained from the original model's architecture
        fusion_layers = {4: self.fusion_p3, 6: self.fusion_p4, 9: self.fusion_p5}
        self.hook_handles = []

        def make_hook(fusion_engine):
            def hook(module, input, output):
                # output is the feature map (tensor)
                # The task embedding is stored in the forward context; we need to access it.
                # We'll store the current task embedding as an attribute before forward.
                task_emb = self.current_task_embedding
                return fusion_engine(output, task_emb)
            return hook

        for idx, fusion_engine in fusion_layers.items():
            layer = self.core_model.model[idx]
            handle = layer.register_forward_hook(make_hook(fusion_engine))
            self.hook_handles.append(handle)

    def forward(self, x, task_id):
        # Retrieve embedding for the given task
        task_emb = self.task_embeddings(task_id)   # (B,384)
        # Store it so hooks can access it
        self.current_task_embedding = task_emb

        # Run the original YOLO forward. Hooks will modify the feature maps.
        # The output is exactly what the original DetectionModel returns
        output = self.core_model(x)
        # Clean up
        del self.current_task_embedding
        return output