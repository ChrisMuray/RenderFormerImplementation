import torch
from torch import nn, Tensor
import torch.nn.functional as F
from scene_model import Camera

# Ray Bundle Embedding. Each ray bundle is a collection of 8 × 8
# rays that go through the center of the pixels of the corresponding
# pixel patch. Because the scene is expressed in camera coordinates
# in the view-dependent stage, the origin of all rays is (0, 0, 0). We
# therefore, only need to encode the normalized directions of each
# ray. We stack the 64 direction vectors in a 192-dimensional vector
# which is subsequently encoded by a single linear layer followed by
# RMS-Normalization into a 768-dimensional token.

def get_ray_bundles_from_camera(cam: Camera, size_px: int, patch_size_px: int):
    """
    Computes the ray bundles for a camera.
    
    Args:
    camera: Camera from a Scene

    Returns:
    rays: (#bundles, bundle_size_px^2) IN CAMERA COORDINATES
    """
    if (size_px % patch_size_px != 0):
        raise ValueError("Image size must be a multiple of bundle size")
    
    # Working in camera coordinates
    forward = Tensor([0, 0, 1])
    up = Tensor([0, -1, 0])
    right = Tensor([1, 0, 0])

    img_plane_size = 2 * torch.tan(torch.deg2rad(Tensor([cam.fov])/2))
    
    points = (torch.arange(size_px)+0.5)/size_px - 0.5
    u, v = torch.meshgrid(points, points, indexing='xy')
    
    rays = F.normalize(forward + img_plane_size * (u.reshape(-1,1) * right + v.reshape(-1,1) * up))

    num_patches = size_px // patch_size_px
    rays = rays.reshape(num_patches, patch_size_px, num_patches, patch_size_px, 3)\
        .permute(0, 2, 1, 3, 4)\
        .reshape(-1, patch_size_px**2, 3)

    return rays

# class RayGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, camera: Camera) -> Tensor:
#         return torch.Tensor()
        

# class ViewDependentTransformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cross_attn = 0 #TODO
#         self.self_attn = 0 #TODO
#         self.ffn = 0 #TODO
    
#     def forward(self, ray_bundle_tokens):