import numpy as np
import torch
from typing import Callable, Optional
from sklearn.decomposition import NMF
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def dff(activations: np.ndarray, n_components: int = 5):
    """
    Deep Feature Factorization: returns concepts (W) and explanation maps (H).
    No heatmap processing here.
    """
    batch_size, channels, h, w = activations.shape
    reshaped_activations = activations.transpose((1, 0, 2, 3))
    reshaped_activations[np.isnan(reshaped_activations)] = 0
    reshaped_activations = reshaped_activations.reshape(
        reshaped_activations.shape[0], -1)
    offset = reshaped_activations.min(axis=-1)
    reshaped_activations = reshaped_activations - offset[:, None]

    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(reshaped_activations)
    H = model.components_
    concepts = W + offset[:, None]
    explanations = H.reshape(n_components, batch_size, h, w)
    explanations = explanations.transpose((1, 0, 2, 3))
    return concepts, explanations  # no scaling


class DFFExtractor:
    """
    Extracts only concepts and explanation maps from a model,
    skipping all visualization/postprocessing (no scaling, no labels).
    """

    def __init__(self,
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 reshape_transform: Optional[Callable] = None):
        self.model = model
        self.activations_and_grads = ActivationsAndGradients(
            self.model, [target_layer], reshape_transform)

    def __call__(self,
                 input_tensor: torch.Tensor,
                 n_components: int = 16):
        _ = self.activations_and_grads(input_tensor)
        with torch.no_grad():
            activations = self.activations_and_grads.activations[0].cpu().numpy()
        concepts, explanations = dff(activations, n_components=n_components)
        return concepts, explanations  # raw form, untouched

    def release(self):
        self.activations_and_grads.release()

    def __del__(self):
        self.release()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.release()
        return isinstance(exc_value, IndexError)  # if handled, suppress


# Example usage
def extract_dff_features(model: torch.nn.Module,
                         target_layer: torch.nn.Module,
                         input_tensor: torch.Tensor,
                         reshape_transform: Optional[Callable] = None,
                         n_components: int = 8):
    extractor = DFFExtractor(model, target_layer, reshape_transform)
    concepts, explanations = extractor(input_tensor[None, :], n_components)
    extractor.release()
    return concepts, explanations
