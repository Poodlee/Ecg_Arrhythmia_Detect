import torch.nn.functional as F
import torch
import numpy as np
# Gradient-weighted Class Activation Mapping
class GradCam:
    def __init__(self, network, target_layer):
        """
        Initialize Grad-CAM with the skorch NeuralNetClassifier and the target layer.
        """
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

    def generate_heatmap(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        """
        self.net.zero_grad()  # Reset gradients on the underlying PyTorch model
        output = self.net(**x)  # Pass input through the PyTorch model

        if class_idx is None:
            class_idx = torch.argmax(output)

        # Backpropagate to get gradients
        class_score = output[:, class_idx][0]
        class_score.backward()

        # Compute Grad-CAM
        gradients = self.gradients.detach()
        feature_maps = self.feature_maps.detach()
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                
        cam = torch.sum(weights * feature_maps, dim=1).squeeze() 
        
        if F.relu(cam).sum()!=0:
            cam = F.relu(cam)  # Apply ReLU to focus on positive contributions 
        else:
            cam = 1-cam
        #plt.imshow(cam.to('cpu'))
        
        # Normalize heatmap        
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    @staticmethod
    def overlay_heatmap(image, heatmap, alpha=0.3, cmap='jet'):
        """
        Overlay the Grad-CAM heatmap on the original image.
        """
        import cv2
    
        # Convert heatmap to numpy array
        heatmap = heatmap.cpu().numpy()        
        image = image.squeeze().cpu().numpy()  
        # Resize heatmap to match the input image dimensions
        #heatmap_resized = cv2.resize(heatmap, (image.shape[2], image.shape[1]))  # Resize to (H, W)
        heatmap_resized = cv2.resize(heatmap, (120, 120))  # Resize to (H, W)                
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_resized = np.where(heatmap_resized>100,heatmap_resized,heatmap_resized-10)
        
        #heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)  # Apply a colormap
        
        a, b = image.min(), image.max()        
        #heatmap_resized = normalize_tensor(heatmap_resized, a, b)                        
        # Normalize and convert image to range 0-255
        if len(image.shape) == 2:  # If grayscale
            #print('GRAYSCALE')
            image = np.uint8(255 * (image - image.min()) / (image.max() - image.min()))
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel (BGR)
        else:  # If RGB or multi-channel
            image = np.uint8(255 * (image - image.min()) / (image.max() - image.min()))
    
        # Blend the heatmap with the original image
                    
        #plt.imshow(heatmap_colored)
        #overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)       
        # image = np.transpose(image, (0,2,3,1))         
        return image[0] + heatmap_resized*alpha
