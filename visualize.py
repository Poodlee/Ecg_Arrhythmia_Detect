import torch.nn.functional as F
import torch
import numpy as np
# Gradient-weighted Class Activation Mapping
class GradCam:
    def __init__(self, network, target_layer):
        self.net = network
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_feature_maps)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        """
        Save the feature maps during the forward pass.
        """
        self.feature_maps = output

    def _save_gradients(self, module, grad_input, grad_output):
        """
        Save the gradients during the backward pass.
        """
        self.gradients = grad_output[0]

    def generate_heatmap(self, x, class_idx=None, sample_idx=0):
        """
        Generate Grad-CAM heatmap.
        """
        self.net.zero_grad()  # Reset gradients on the underlying PyTorch model
        output = self.net(**x)  # Extract feature map 

        if class_idx is None:
            class_idx = torch.argmax(output[sample_idx]) # First sample's class index

        # Backpropagate to get gradients
        class_score = output[sample_idx, class_idx]
        class_score.backward(retain_graph=True) # Extract gradient

        # Compute Grad-CAM
        gradients = self.gradients[sample_idx].detach()  # shape: (C, H, W)
        feature_maps = self.feature_maps[sample_idx].detach()  # shape: (C, H, W)
        
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)        
        cam = torch.sum(weights * feature_maps, dim=0) 
        
        cam = F.relu(cam) if F.relu(cam).sum() != 0 else 1 - cam
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize
        return cam

    @staticmethod
    def overlay_heatmap(image, heatmap, alpha=0.3, cmap='jet'):
        """
        Overlay the Grad-CAM heatmap on the original image.
        """
        import cv2
        print(f"[DEBUG] Original image shape (after squeeze): {image.shape}")
        print(f"[DEBUG] Heatmap shape: {heatmap.shape}")

        image = image.squeeze().cpu().numpy()  
    
        # Normalize image to [0, 255]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = np.uint8(255 * image)
        
        heatmap = heatmap.cpu().numpy()        
            
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)

        # Apply color map
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                              
        # Convert grayscale image to 3-channel if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        print(f"[DEBUG] Final image shape: {image.shape}")
        print(f"[DEBUG] Final heatmap_colored shape: {heatmap_colored.shape}")

        # Blend the heatmap with the original image
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)       
        return overlayed
