import json
from scene_model import Scene, Camera
from renderformer import get_ray_bundles_from_camera
import torch
import matplotlib
import numpy as np

with open('examples/cbox-bunny.json', 'r') as f:
    data = json.load(f)
cam: Camera = Scene(**data).cameras[0]
cam.fov=60

ray_ends = get_ray_bundles_from_camera(cam, 16, 4)

print(ray_ends.shape)

# plot them
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Origin
origin = torch.zeros(3)

# Draw lines from origin to each point

num_bundles = len(ray_ends)
colors = lambda i: matplotlib.colormaps['turbo'](i / num_bundles)

for i, bundle in enumerate(ray_ends):
    color = colors(i)
    for p in bundle:
        xs, ys, zs = zip(origin, p)
        ax.plot(xs, ys, zs, color=color)
    bundle_np = bundle.cpu().numpy() if hasattr(bundle, 'cpu') else np.array(bundle)
    ax.scatter(bundle_np[:, 0], bundle_np[:, 1], bundle_np[:, 2], color=[color])
# Axes settings

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title("Camera rays")
plt.savefig('rays.png')