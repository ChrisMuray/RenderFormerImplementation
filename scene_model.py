from pydantic import BaseModel
from typing import Optional, Dict

class Transform(BaseModel):
    translation: list[float]
    rotation: list[float]
    scale: list[float]
    normalize: bool

class Material(BaseModel):
    diffuse: list[float]
    specular: list[float]
    random_diffuse_max: float
    roughness: float
    emissive: list[float]
    smooth_shading: bool
    rand_tri_diffuse_seed: Optional[float]

class SceneObject(BaseModel):
    mesh_path: str
    transform: Transform
    material: Material

class Camera(BaseModel):
    position: list[float]
    look_at: list[float]
    up: list[float]
    fov: float

class Scene(BaseModel):
    scene_name: str
    objects: Dict[str, SceneObject]
    cameras: list[Camera]

### Usage
# with open('examples/cbox-bunny.json', 'r') as f:
#     data = json.load(f)
# scene = Scene(**data)


        