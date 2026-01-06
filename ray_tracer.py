import argparse
from turtle import color
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
import light
from material import Material
from scene_settings import SceneSettings
import surfaces
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

#######################   helper functions   ###########################


def organize_scene(objects):
    obj_lst = []
    lights = []
    materials = []

    for obj in objects:
        if isinstance(obj, (Sphere, InfinitePlane, Cube)):
            obj_lst.append(obj)
        elif isinstance(obj, Light):
            lights.append(obj)
        elif isinstance(obj, Material):
            materials.append(obj)

    return obj_lst, lights, materials



###########################   GEOMETRY   ###############################

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

#------------------------------------------------------------

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

    pos = np.array(cube.position, dtype=float)
    # calculate the bounds of the cube
    half = cube.scale / 2.0
    # left lower back corner 
    min_bound = pos - half
    # right upper front corner
    max_bound = pos + half

    # ray-box intersection points
    # t_near is the largest entering t value from all axis
    # t_far is the smallest exiting t value from all axis
    t_near = -np.inf
    t_far = np.inf
    hit_normal = None

    for i in range(3):  # x, y, z
        if abs(ray_direction[i]) < 1e-6: # Ray is not moving in this axis
            # Check if the ray origin is inside the cube bounds on this axis
            if ray_origin[i] < min_bound[i] or ray_origin[i] > max_bound[i]:
                return None
        else:
            # find t for when the ray crosses the two planes on this axis
            # solution for: ray_origin[i] + t * ray_direction[i] = min_bound[i]
            t1_i = (min_bound[i] - ray_origin[i]) / ray_direction[i]
            t2_i = (max_bound[i] - ray_origin[i]) / ray_direction[i]

            # swap t1 and t2 if necessary
            t_entering_i = min(t1_i, t2_i)
            t_exiting_i = max(t1_i, t2_i)

            if t_entering_i > t_near:
                t_near = t_entering_i
                # update hit normal, box is axis-aligned so normal is determined by entering axis to the box and the entering side
                hit_normal = np.zeros(3)
                hit_normal[i] = -1 if t1_i > t2_i else 1
            t_far = min(t_far, t_exiting_i)

            # if at this point the largest entering t is larger than the smallest exiting t, no intersection, the ray misses the box
            if t_near > t_far:
                return None

    # box is behind the ray
    if t_far < 1e-6:
        return None

    # box survived tests, intersection occurs at t_near
    # ray can be inside the box, so we take the nearest positive t
    t_hit = t_near if t_near > 1e-6 else t_far
    hit_point = ray_origin + t_hit * ray_direction

    # ensure normal points against the ray
    if np.dot(hit_normal, ray_direction) > 0:
        hit_normal = -hit_normal

    return t_hit, hit_point, hit_normal   

#-----------------------------------------------------------

def closest_intersection(ray_origin, ray_direction, obj_lst):
    closest_t = np.inf
    closest_hit_details = None

    for obj in obj_lst:
        result = None

        if isinstance(obj, Sphere):
            result = intersect_sphere(ray_origin, ray_direction, obj)
        
        elif isinstance(obj, InfinitePlane):
            result = intersect_plane(ray_origin, ray_direction, obj)

        elif isinstance(obj, Cube):
            result = intersect_cube(ray_origin, ray_direction, obj)

        # if there was an intersection - check if closest
        if result is not None:
            # unpack result
            t, hit_point, hit_normal = result
            if t < closest_t:
                closest_t = t
                closest_hit_details = (t, hit_point, hit_normal, obj)

    return closest_hit_details

###########################   SHADOWING   ###############################

def build_light_plane(light_direction):
    # remember: light direction is from point to light
    # we can’t uniquely define a perpendicular vector to a given normal    # so choose arbitrary vector w that is not parallel to light_direction
    # so we choose arbitrary (not random) vector w that is not parallel to light_direction
    
    # math trick to choose w that is not parallel to light_direction
    # if light_direction is mostly not aligned with x axis, choose x axis as w
    if abs(light_direction[0]) < 0.9:
        w = np.array([1, 0, 0])
    else:
    # choose y axis as w
        w = np.array([0, 1, 0])

    # find u that u = light_direction × w
    u = np.cross(light_direction, w)
    u = u / np.linalg.norm(u)

     # find v that v = light_direction × u
    v = np.cross(light_direction, u)
    v = v / np.linalg.norm(v)

    return u, v


#-----------------------------------------------------------


def soft_shadow(point_on_obj, light_src, obj_lst, shadow_rays_num):
    # shadow_rays is given by scene settings

    light_vec = light_src.position - point_on_obj
    light_length = np.linalg.norm(light_vec)
    light_dir_norm = light_vec / light_length

    u, v = build_light_plane(light_dir_norm)

    rays_reaching_light = 0

    shadow_rays_num = int(shadow_rays_num)
    total_shadow_rays = shadow_rays_num * shadow_rays_num

    # itarate over a grid of shadow_rays x shadow_rays
    for i in range(shadow_rays_num):
        for j in range(shadow_rays_num):
            # choose random point inside sub-square
            # translating the grid to be 1*1 square centered at light position
            # np.random.rand() E [0, 1)
            ru = (i + np.random.rand()) / shadow_rays_num - 0.5
            rv = (j + np.random.rand()) / shadow_rays_num - 0.5

            # now (ru, rv) is a random point in the light square of size 1*1 centered at light position
            # build the vector in the light plane the points from the center of the light to the random point inside the sub-square of the grid
            offset_vector = (ru * u + rv * v) * light_src.radius # vector = direction * length
            new_light_src_position = light_src.position + offset_vector

            new_shadow_ray = new_light_src_position - point_on_obj
            dist = np.linalg.norm(new_shadow_ray)
            new_shadow_ray_norm = new_shadow_ray / dist

            # offset the shadow ray origin to avoid self-intersection
            shadow_origin = point_on_obj + 1e-4 * new_shadow_ray_norm # epsilon for numerical errors
            hit = closest_intersection(shadow_origin, new_shadow_ray_norm, obj_lst)

            if hit is None: # no intersection - ray reached the light
                rays_reaching_light += 1
            else:
                t, hit_point, hit_normal, obj = hit
                if t > dist: # intersection is beyond the light source
                    rays_reaching_light += 1

    ratio = rays_reaching_light / total_shadow_rays

    # light always exists but the shadow reduces its intensity
    return (1 - light_src.shadow_intensity) + light_src.shadow_intensity * ratio


###########################   LIGHTING   ###############################

def compute_diffuse(material, light, normal, light_dir):
    # Idiff​=Kd​⋅Ip​⋅cos(θ)=Kd​⋅Ip​⋅(N⋅L)
    # Kd = material.diffuse_color
    # Ip = light.color
    # N = normal
    # L = light_dir (normalized)
    # cos(θ) = N . L

    # diffuse cant be negative
    N_dot_L = max(0.0, np.dot(normal, light_dir))

    Kd = np.array(material.diffuse_color, dtype=float)
    Ip = np.array(light.color, dtype=float)

    return Kd * Ip * N_dot_L



#-----------------------------------------------------------


def compute_specular(material, light, normal, light_dir, view_dir):
    # Ispec​=Ks​⋅Ip​⋅cosn(ϕ)=Ks​⋅Ip​⋅(R⋅V)n
    # Ks = material.specular_color
    # Ip = light.color
    # R = reflection direction of light_dir around normal
    # V = view_dir (normalized)
    # n = material.shininess
    # cos(ϕ) = R . V

    # R=2(N⋅L)N−L
    R = 2 * np.dot(normal, light_dir) * normal - light_dir
    R_norm = R / np.linalg.norm(R)

    # specular cant be negative
    R_dot_V = max(0.0, np.dot(R_norm, view_dir))

    Ks = np.array(material.specular_color, dtype=float)
    Ip = np.array(light.color, dtype=float)

    return Ks * Ip * (R_dot_V ** material.shininess)

#-----------------------------------------------------------

def compute_lighting(point_on_obj, normal, view_dir, material, lights,obj_lst, scene_settings):

    # RGB color initialized to black
    color = np.zeros(3)

    for light in lights:
        # build light ray
        light_vec = light.position - point_on_obj
        light_length = np.linalg.norm(light_vec)
        light_dir_norm = light_vec / light_length

        diffuse = compute_diffuse(material, light, normal, light_dir_norm)
        specular = compute_specular(material, light, normal, light_dir_norm, view_dir)
        specular *= light.specular_intensity
        soft_shadow_factor = soft_shadow(point_on_obj,light,obj_lst,scene_settings.root_number_shadow_rays)

        color += soft_shadow_factor * (diffuse + specular)

    # color values range correction to [0, 1]
    for i in range(3):
        if color[i] < 0:
            color[i] = 0
        elif color[i] > 1:
            color[i] = 1

    return color


###########################   REFLECTION   ###############################

def reflect(direction, normal):
    # calc return ray
    # R = V - 2(V⋅N)N
    return direction - 2 * np.dot(direction, normal) * normal

###########################   RAY TRACER   ###############################

def ray_tracer(camera, scene_settings, objects, image_width, image_height):
    # split scene objects
    obj_lst, lights, materials = organize_scene(objects)

    # render image
    image = render_scene(camera,scene_settings,obj_lst,lights,materials,image_width,image_height)

    return image

#-----------------------------------------------------------

def rec_ray_tracer(ray_origin, ray_direction, depth,scene_settings, obj_lst, lights, materials):

    # stopping condition
    if depth <= 0:
        return np.zeros(3) # black

    hit = closest_intersection(ray_origin, ray_direction, obj_lst)

    # ray missed all objects, therefore return background color
    if hit is None:
        return np.array(scene_settings.background_color, dtype=float)

    t, hit_point, normal, surface = hit
    material = materials[surface.material_index-1]

    # specular depends on the view direction
    # calc view direction (towards camera)
    # ray direction is from camera to point, so view direction is the opposite
    view_dir = -ray_direction
    view_dir_norm = view_dir / np.linalg.norm(view_dir)

    # calc lighting = diffuse + specular + soft shadow
    color = compute_lighting(hit_point,normal,view_dir_norm, material,lights,obj_lst,scene_settings)

    # calc reflection
    if np.any(material.reflection_color):
        reflect_dir = reflect(ray_direction, normal)
        reflect_dir_norm = reflect_dir / np.linalg.norm(reflect_dir)
        reflect_origin = hit_point + 1e-4 * reflect_dir_norm

        reflected_color = rec_ray_tracer(
            reflect_origin,
            reflect_dir_norm,
            depth - 1,
            scene_settings,
            obj_lst,
            lights,
            materials
        )

        Kr = np.array(material.reflection_color, dtype=float)
        color += Kr * reflected_color


    # calc transparency
    if material.transparency > 0:
        trans_origin = hit_point + 1e-4 * ray_direction

        transparent_color = rec_ray_tracer(
            trans_origin,
            ray_direction,
            depth - 1,
            scene_settings,
            obj_lst,
            lights,
            materials
        )

        color = (
            (1 - material.transparency) * color +
            material.transparency * transparent_color
        )

    # color values range correction to [0, 1]
    for i in range(3):
        if color[i] < 0:
            color[i] = 0
        elif color[i] > 1:
            color[i] = 1

    return color


###########################   RENDER SCENE   ###############################

def render_scene(camera, scene_settings,obj_lst, lights, materials,image_width, image_height):
    # up - a vector from the center of the screen to the top of the screen
    # right - a vector from the center of the screen to the right of the screen


    # image array with RGB values - pixels resolution
    image = np.zeros((image_height, image_width, 3))

    # camera parameters
    cam_pos = np.array(camera.position)
    look_at = np.array(camera.look_at)
    up_vec = np.array(camera.up_vector)

    # camera basis vectors
    forward = look_at - cam_pos
    forward_norm = forward / np.linalg.norm(forward)

    image_right = np.cross(up_vec, forward_norm)
    image_right_norm = image_right / np.linalg.norm(image_right)

    image_up = np.cross(image_right_norm, forward_norm)
    image_up_norm = image_up / np.linalg.norm(image_up)

    # screen geometry
    screen_dist = camera.screen_distance
    screen_width = camera.screen_width
    aspect_ratio = image_width / image_height
    screen_height = screen_width / aspect_ratio

    screen_center = cam_pos + forward_norm * screen_dist

    # iterate over pixels
    for y in range(image_height):
        for x in range(image_width):

            # normalized pixel coordinates in range [-0.5, 0.5]
            px = (x + 0.5) / image_width - 0.5
            py = (y + 0.5) / image_height - 0.5


            # pixel position on the screen
            pixel_pos = (
                screen_center +
                px * screen_width * image_right_norm +
                py * screen_height * image_up_norm
            )

            # build ray
            ray_dir = pixel_pos - cam_pos
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            # trace ray
            color = rec_ray_tracer(
                cam_pos,
                ray_dir,
                scene_settings.max_recursions,
                scene_settings,
                obj_lst,
                lights,
                materials
            )

            # store color (convert to [0,255])
            image[y, x] = color * 255

    return image


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
    image_array = ray_tracer(
        camera,
        scene_settings,
        objects,
        args.width,
        args.height
    )

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
