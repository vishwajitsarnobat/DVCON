# We had 2 options, either clone ultralytics YOLOv8 repo and make modification in the architecture as needed
# OR make a torch wrapper for accessing the different components of the model without touching the main YOLO model.
# We are proceeding with second approach here.

import torch
import torch.nn as nn
from ultralytics import YOLO

from prune_utils import apply_2_4_sparsity

class TaskGuidedFusionEngine(nn.Module):
    '''
    This is for multiplying task embedding and necessary layers from yolo (one layer at a time) and match the dims
    '''
    def __init__(self,feature_channels,embedding_dim=384):
        super().__init__()
        self.channel_projection=nn.Linear(embedding_dim,feature_channels) # defining the layer
        self.sigmoid=nn.Sigmoid() # creates mask of values between 0 and 1, scaling

    def forward(self,x,task_embedding):
        """
        x: Feature map from YOLOv8n backbone: (B,C,H,W)
        task_embedding: (B,384)
        """
        # (B,384) -> (B,C)
        projected_task=self.channel_projection(task_embedding)
        # (B,C) -> (B,C,1,1)
        projected_task=projected_task.unsqueeze(-1).unsqueeze(-1)
        
        # in hardware, values<threshold will be 0, here we keep it continuous for differentiability.
        mask=self.sigmoid(projected_task)
        gated_features=x*mask # broadcasting will make (B,C,1,1) of mask to be (B,C,H,W)
        return gated_features

class TaskAwareYOLO(nn.Module):
    '''
    Actual flow of data in yolo, actual image goes in, (p3,p4,p5) has features at different scale, task embedding is multiplied by them.
    Data goes through all the layers and not just (p3,p4,p5) and gives out final output x.
    '''
    def __init__(self, embedding_path='task_embeddings.pt'):
        super().__init__()

        # .yaml to initialize fresh architecture
        print("Loading base YOLOv8n model...")
        yolo_wrapper = YOLO('yolov8n.pt') # loading pretrained model instead of yaml file
        self.core_model = yolo_wrapper.model  
        
        # Extract the actual sequential list of layers inside the DetectionModel
        self.layers = self.core_model.model
        
        # YOLOv8 keeps a master list of which layers need to be saved for skip connections
        self.save_layers = self.core_model.save  
        
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        raw_embeddings = torch.load(embedding_path, map_location=device)
        # lookup table using indexing
        self.task_embeddings=nn.Embedding.from_pretrained(raw_embeddings,freeze=True) # (num_embeddings,embedding_dim) i.e. (14,384)
        # freeze=True is needed to avoid grad updates during backward pass

        # just initialising the objects, used later in forward
        self.fusion_p3=TaskGuidedFusionEngine(feature_channels=64,embedding_dim=384)
        self.fusion_p4=TaskGuidedFusionEngine(feature_channels=128,embedding_dim=384)
        self.fusion_p5=TaskGuidedFusionEngine(feature_channels=256,embedding_dim=384)

    def forward(self,x,task_id):
        """
        Input image tensor (B,3,640,640), 3 for RGB
        """
        y=[]
        task_id=task_id.to(self.task_embeddings.weight.device)
        task_emb=self.task_embeddings(task_id)

        for i, m in enumerate(self.layers):
            # If the layer needs a specific previous layer (not just the immediate last one)
            if m.f != -1:
                if isinstance(m.f, list):  # Needs multiple layers (Concat)
                    x = [x if j == -1 else y[j] for j in m.f]
                else:                      # Needs one specific layer
                    x = y[m.f]
            
            # Run the layer
            x = m(x)  
            
            # Mask P3 features
            if i == 4:
                x = self.fusion_p3(x, task_emb)
            # Mask P4 features
            elif i == 6:
                x = self.fusion_p4(x, task_emb)
            # Mask P5 features
            elif i == 9:
                x = self.fusion_p5(x, task_emb)

            # Save the output to our 'y' list ONLY if a future layer needs it, to save memory
            y.append(x if i in self.save_layers else None)
            
        return x # The final layer returns the actual bounding box predictions

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print("\nInitializing TaskAwareYOLO...")
    model=TaskAwareYOLO(embedding_path='task_embeddings.pt').to(device)
    print("\nApplying 2:4 structured sparsity")
    model=apply_2_4_sparsity(model)

    # turns off dropout, stabalizes batch norm
    model.eval() # specifically in pytorch, this switches from training mode to eval mode

    dummy_img=torch.randn(1,3,640,640).to(device)
    dummy_task_id=torch.tensor([3]).to(device)

    print("\nRunning forward pass...")
    with torch.no_grad(): # no gradients needed for testing
        try:
            output=model(dummy_img,dummy_task_id)
            print("TEST SUCCESS! Data flowed through the network without crashing.")
            
            # the output can be tuple, depending on the head configuration
            if isinstance(output,tuple):
                print("Head returns tuple!")
                main_output=output[0]
            else:
                print("Head returns normal tensor!")
                main_output=output
            
            print(f"Final output shape: {main_output.shape}")
            print("Meaning: (Batch Size, [Bounding box coordinates + class scores], Anchors)")
        except Exception as e:
            print(f"\nCRASH: PyTorch threw an error:")
            print(e)

if __name__=="__main__":
    main()