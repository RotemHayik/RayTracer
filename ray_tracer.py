import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

#########################################################

## math functions ##

## def normalize(v) = v / np.linalg.norm(v)

## def dot(a, b) = np.dot(a, b)

## def cross(a, b) = np.cross(a, b)

## def distance(a, b) = np.linalg.norm(a - b)

## def length(v) = np.linalg.norm(v)

##########################################################

## helper functions ##

def intersect_sphere(ray_origin, ray_direction, sphere):
    ## ray origin = P0
    ## ray direction = V (normalized)

    ## the geometric method for ray-sphere intersection

    # L = O - P0
    L = sphere.position - ray_origin

    # t_ca = L . V
    t_ca = np.dot(L, ray_direction)
    if t_ca < 1e-6:
        return None # sphere is behind the ray

    # d^2 = L . L - t_ca^2
    sqrt_d = np.dot(L, L) - t_ca**2
    sqrt_r = sphere.radius ** 2
    if sqrt_d > sqrt_r:
        return None # no intersection

    # intersections distances along the ray
    t_hc = np.sqrt(sqrt_r - sqrt_d)
    t1 = t_ca - t_hc
    t2 = t_ca + t_hc

    t = None
    if t1 > 1e-6 and t2 > 1e-6:
        t = min(t1, t2) # returns closest intersection
    elif t1 > 1e-6:
        t = t1 # returns the intersection in front of the ray
    elif t2 > 1e-6:
        t = t2 # returns the intersection in front of the ray
    else:
        return None

    intersection = ray_origin + t * ray_direction

    # N = (P - O) / ||P - O||
    normal = intersection - sphere.position
    normal = normal / np.linalg.norm(normal)

    return t, intersection, normal

#-----------------------------------------------------------

def intersect_plane(ray_origin, ray_direction, plane):

    # Normalizing the plane normal to ensure accurate calculations
    N = np.array(plane.normal, dtype=float)
    N = N / np.linalg.norm(N)   
    c = plane.offset

    V_dot_N = np.dot(ray_direction, N)
    if abs(V_dot_N) < 1e-6: # "equal to zero" - ray is parallel to the plane
        return None  # no intersection

    # place the ray origin in the plane equation
    # find t in the ray equation = t = (c - P0 . N) / (V . N)
    t = (c - np.dot(ray_origin, N)) / V_dot_N
    if t < 1e-6:
        return None # plane is behind the ray

    hit_point = ray_origin + t * ray_direction

    # ensure the normal is against the ray direction
    #the normal and the ray create an obtuse angle
    if np.dot(ray_direction, N) > 0:
        N = -N

    return t, hit_point, N

#-----------------------------------------------------------

def intersect_cube(ray_origin, ray_direction, cube):
    half = cube.scale / 2.0

    # גבולות הקובייה
    min_bound = cube.position - half
    max_bound = cube.position + half

    t_near = -np.inf
    t_far = np.inf
    hit_normal = None

    for i in range(3):  # x, y, z
        if abs(ray_direction[i]) < 1e-6:
            # הקרן מקבילה למישורי הציר
            if ray_origin[i] < min_bound[i] or ray_origin[i] > max_bound[i]:
                return None
        else:
            t1 = (min_bound[i] - ray_origin[i]) / ray_direction[i]
            t2 = (max_bound[i] - ray_origin[i]) / ray_direction[i]

            t1_axis = min(t1, t2)
            t2_axis = max(t1, t2)

            if t1_axis > t_near:
                t_near = t1_axis
                hit_normal = np.zeros(3)
                hit_normal[i] = -1 if t1 > t2 else 1

            t_far = min(t_far, t2_axis)

            # תנאי פספוס לפי השקופית
            if t_near > t_far:
                return None

    # תנאי: הקובייה מאחורי הקרן
    if t_far < 1e-6:
        return None

    # נקודת פגיעה היא ב־t_near
    t_hit = t_near if t_near > 1e-6 else t_far
    hit_point = ray_origin + t_hit * ray_direction

    return t_hit, hit_point, hit_normal   

#-----------------------------------------------------------

def closest_intersection(ray, surfaces):
    # Find the closest intersection of the ray with the surfaces
    pass

#########################################################

## ray tracer logic finctions ##



#########################################################

## ray tracer main function ##

def trace_ray(ray_origin, ray_direction, depth):
    # Ray tracing logic
    pass

def render_scene(camera, scene_settings, surfaces, lights, materials, image_width, image_height):
    # Scene rendering logic
    pass

#########################################################


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer

    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
