import argparse
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
import time
from numba import njit

from camera import Camera
from light import Light
import light
from material import Material
from scene_settings import SceneSettings
import surfaces
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

EPSILON = 1e-4


###########################   NUMBA JIT FUNCTIONS   ###############################

@njit(cache=True)
def _intersect_sphere_jit(ray_origin, ray_direction, sphere_pos, sphere_radius):
    """JIT-compiled sphere intersection. Returns (t, hit_x, hit_y, hit_z, nx, ny, nz) or (-1, 0,0,0,0,0,0) if no hit."""
    # L = O - P0
    L = sphere_pos - ray_origin

    # t_ca = L . V
    t_ca = np.dot(L, ray_direction)

    # d^2 = L . L - t_ca^2
    d_squared = np.dot(L, L) - t_ca * t_ca
    r_squared = sphere_radius * sphere_radius

    if d_squared > r_squared:
        return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # intersections distances along the ray
    t_hc = np.sqrt(r_squared - d_squared)
    t1 = t_ca - t_hc
    t2 = t_ca + t_hc

    t = -1.0
    if t1 > EPSILON and t2 > EPSILON:
        t = min(t1, t2)
    elif t1 > EPSILON:
        t = t1
    elif t2 > EPSILON:
        t = t2
    else:
        return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # intersection point
    hit_x = ray_origin[0] + t * ray_direction[0]
    hit_y = ray_origin[1] + t * ray_direction[1]
    hit_z = ray_origin[2] + t * ray_direction[2]

    # normal
    nx = hit_x - sphere_pos[0]
    ny = hit_y - sphere_pos[1]
    nz = hit_z - sphere_pos[2]
    n_len = np.sqrt(nx*nx + ny*ny + nz*nz)
    nx /= n_len
    ny /= n_len
    nz /= n_len

    return t, hit_x, hit_y, hit_z, nx, ny, nz


@njit(cache=True)
def _intersect_plane_jit(ray_origin, ray_direction, plane_normal, plane_offset):
    """JIT-compiled plane intersection. Returns (t, hit_x, hit_y, hit_z, nx, ny, nz) or (-1, ...) if no hit."""
    # Normalize plane normal
    n_len = np.sqrt(plane_normal[0]**2 + plane_normal[1]**2 + plane_normal[2]**2)
    nx = plane_normal[0] / n_len
    ny = plane_normal[1] / n_len
    nz = plane_normal[2] / n_len

    V_dot_N = ray_direction[0]*nx + ray_direction[1]*ny + ray_direction[2]*nz
    if abs(V_dot_N) < EPSILON:
        return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    P0_dot_N = ray_origin[0]*nx + ray_origin[1]*ny + ray_origin[2]*nz
    t = (plane_offset - P0_dot_N) / V_dot_N

    if t < EPSILON:
        return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    hit_x = ray_origin[0] + t * ray_direction[0]
    hit_y = ray_origin[1] + t * ray_direction[1]
    hit_z = ray_origin[2] + t * ray_direction[2]

    # Ensure normal points against ray
    if V_dot_N > 0:
        nx, ny, nz = -nx, -ny, -nz

    return t, hit_x, hit_y, hit_z, nx, ny, nz


@njit(cache=True)
def _intersect_cube_jit(ray_origin, ray_direction, cube_pos, cube_scale):
    """JIT-compiled cube intersection."""
    half = cube_scale / 2.0
    min_x, min_y, min_z = cube_pos[0] - half, cube_pos[1] - half, cube_pos[2] - half
    max_x, max_y, max_z = cube_pos[0] + half, cube_pos[1] + half, cube_pos[2] + half

    t_near = -1e30
    t_far = 1e30
    hit_nx, hit_ny, hit_nz = 0.0, 0.0, 0.0

    # X axis
    if abs(ray_direction[0]) < EPSILON:
        if ray_origin[0] < min_x or ray_origin[0] > max_x:
            return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        t1 = (min_x - ray_origin[0]) / ray_direction[0]
        t2 = (max_x - ray_origin[0]) / ray_direction[0]
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_near:
            t_near = t1
            hit_nx, hit_ny, hit_nz = -1.0 if ray_direction[0] > 0 else 1.0, 0.0, 0.0
        if t2 < t_far:
            t_far = t2
        if t_near > t_far:
            return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Y axis
    if abs(ray_direction[1]) < EPSILON:
        if ray_origin[1] < min_y or ray_origin[1] > max_y:
            return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        t1 = (min_y - ray_origin[1]) / ray_direction[1]
        t2 = (max_y - ray_origin[1]) / ray_direction[1]
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_near:
            t_near = t1
            hit_nx, hit_ny, hit_nz = 0.0, -1.0 if ray_direction[1] > 0 else 1.0, 0.0
        if t2 < t_far:
            t_far = t2
        if t_near > t_far:
            return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Z axis
    if abs(ray_direction[2]) < EPSILON:
        if ray_origin[2] < min_z or ray_origin[2] > max_z:
            return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        t1 = (min_z - ray_origin[2]) / ray_direction[2]
        t2 = (max_z - ray_origin[2]) / ray_direction[2]
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_near:
            t_near = t1
            hit_nx, hit_ny, hit_nz = 0.0, 0.0, -1.0 if ray_direction[2] > 0 else 1.0
        if t2 < t_far:
            t_far = t2
        if t_near > t_far:
            return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if t_far < EPSILON:
        return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    t_hit = t_near if t_near > EPSILON else t_far
    hit_x = ray_origin[0] + t_hit * ray_direction[0]
    hit_y = ray_origin[1] + t_hit * ray_direction[1]
    hit_z = ray_origin[2] + t_hit * ray_direction[2]

    # Ensure normal points against ray
    dot = hit_nx * ray_direction[0] + hit_ny * ray_direction[1] + hit_nz * ray_direction[2]
    if dot > 0:
        hit_nx, hit_ny, hit_nz = -hit_nx, -hit_ny, -hit_nz

    return t_hit, hit_x, hit_y, hit_z, hit_nx, hit_ny, hit_nz


@njit(cache=True)
def _closest_intersection_jit(ray_origin, ray_direction,
                               sphere_positions, sphere_radii, sphere_mat_indices,
                               plane_normals, plane_offsets, plane_mat_indices,
                               cube_positions, cube_scales, cube_mat_indices):
    """JIT-compiled closest intersection across all objects.
    Returns (t, hit_x, hit_y, hit_z, nx, ny, nz, mat_idx, obj_type) or (-1, ...) if no hit.
    obj_type: 0=sphere, 1=plane, 2=cube
    """
    closest_t = 1e30
    best_hx, best_hy, best_hz = 0.0, 0.0, 0.0
    best_nx, best_ny, best_nz = 0.0, 0.0, 0.0
    best_mat_idx = -1
    best_obj_type = -1
    best_obj_idx = -1

    # Test spheres
    num_spheres = sphere_positions.shape[0]
    for i in range(num_spheres):
        t, hx, hy, hz, nx, ny, nz = _intersect_sphere_jit(
            ray_origin, ray_direction, sphere_positions[i], sphere_radii[i])
        if t > 0 and t < closest_t:
            closest_t = t
            best_hx, best_hy, best_hz = hx, hy, hz
            best_nx, best_ny, best_nz = nx, ny, nz
            best_mat_idx = sphere_mat_indices[i]
            best_obj_type = 0
            best_obj_idx = i

    # Test planes
    num_planes = plane_normals.shape[0]
    for i in range(num_planes):
        t, hx, hy, hz, nx, ny, nz = _intersect_plane_jit(
            ray_origin, ray_direction, plane_normals[i], plane_offsets[i])
        if t > 0 and t < closest_t:
            closest_t = t
            best_hx, best_hy, best_hz = hx, hy, hz
            best_nx, best_ny, best_nz = nx, ny, nz
            best_mat_idx = plane_mat_indices[i]
            best_obj_type = 1
            best_obj_idx = i

    # Test cubes
    num_cubes = cube_positions.shape[0]
    for i in range(num_cubes):
        t, hx, hy, hz, nx, ny, nz = _intersect_cube_jit(
            ray_origin, ray_direction, cube_positions[i], cube_scales[i])
        if t > 0 and t < closest_t:
            closest_t = t
            best_hx, best_hy, best_hz = hx, hy, hz
            best_nx, best_ny, best_nz = nx, ny, nz
            best_mat_idx = cube_mat_indices[i]
            best_obj_type = 2
            best_obj_idx = i

    if best_obj_type < 0:
        return -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, -1, -1

    return closest_t, best_hx, best_hy, best_hz, best_nx, best_ny, best_nz, best_mat_idx, best_obj_type, best_obj_idx


# Global variables for multiprocessing workers (set by initializer)
_worker_data = {}

#######################   helper functions   ###########################


def organize_scene(objects):
    """Organize scene objects and pre-cache geometry as numpy arrays."""
    obj_lst = []
    lights = []
    materials = []

    # Separate lists for each object type
    spheres = []
    planes = []
    cubes = []

    for obj in objects:
        if isinstance(obj, Sphere):
            obj_lst.append(obj)
            spheres.append(obj)
        elif isinstance(obj, InfinitePlane):
            obj_lst.append(obj)
            planes.append(obj)
        elif isinstance(obj, Cube):
            obj_lst.append(obj)
            cubes.append(obj)
        elif isinstance(obj, Light):
            lights.append(obj)
        elif isinstance(obj, Material):
            materials.append(obj)

    # Pre-cache geometry as numpy arrays for JIT functions
    if spheres:
        sphere_positions = np.array([s.position for s in spheres], dtype=np.float64)
        sphere_radii = np.array([s.radius for s in spheres], dtype=np.float64)
        sphere_mat_indices = np.array([s.material_index for s in spheres], dtype=np.int32)
    else:
        sphere_positions = np.empty((0, 3), dtype=np.float64)
        sphere_radii = np.empty(0, dtype=np.float64)
        sphere_mat_indices = np.empty(0, dtype=np.int32)

    if planes:
        plane_normals = np.array([p.normal for p in planes], dtype=np.float64)
        plane_offsets = np.array([p.offset for p in planes], dtype=np.float64)
        plane_mat_indices = np.array([p.material_index for p in planes], dtype=np.int32)
    else:
        plane_normals = np.empty((0, 3), dtype=np.float64)
        plane_offsets = np.empty(0, dtype=np.float64)
        plane_mat_indices = np.empty(0, dtype=np.int32)

    if cubes:
        cube_positions = np.array([c.position for c in cubes], dtype=np.float64)
        cube_scales = np.array([c.scale for c in cubes], dtype=np.float64)
        cube_mat_indices = np.array([c.material_index for c in cubes], dtype=np.int32)
    else:
        cube_positions = np.empty((0, 3), dtype=np.float64)
        cube_scales = np.empty(0, dtype=np.float64)
        cube_mat_indices = np.empty(0, dtype=np.int32)

    # Store cached geometry
    geometry_cache = {
        'sphere_positions': sphere_positions,
        'sphere_radii': sphere_radii,
        'sphere_mat_indices': sphere_mat_indices,
        'plane_normals': plane_normals,
        'plane_offsets': plane_offsets,
        'plane_mat_indices': plane_mat_indices,
        'cube_positions': cube_positions,
        'cube_scales': cube_scales,
        'cube_mat_indices': cube_mat_indices,
    }

    return obj_lst, lights, materials, geometry_cache



###########################   GEOMETRY   ###############################

def intersect_sphere(ray_origin, ray_direction, sphere):
    """Wrapper that calls JIT-compiled sphere intersection."""
    sphere_pos = np.asarray(sphere.position, dtype=np.float64)
    result = _intersect_sphere_jit(ray_origin, ray_direction, sphere_pos, sphere.radius)
    if result[0] < 0:
        return None
    t, hx, hy, hz, nx, ny, nz = result
    return t, np.array([hx, hy, hz]), np.array([nx, ny, nz])

#------------------------------------------------------------

def intersect_plane(ray_origin, ray_direction, plane):
    """Wrapper that calls JIT-compiled plane intersection."""
    plane_normal = np.asarray(plane.normal, dtype=np.float64)
    result = _intersect_plane_jit(ray_origin, ray_direction, plane_normal, plane.offset)
    if result[0] < 0:
        return None
    t, hx, hy, hz, nx, ny, nz = result
    return t, np.array([hx, hy, hz]), np.array([nx, ny, nz])

#-----------------------------------------------------------

def intersect_cube(ray_origin, ray_direction, cube):
    """Wrapper that calls JIT-compiled cube intersection."""
    cube_pos = np.asarray(cube.position, dtype=np.float64)
    result = _intersect_cube_jit(ray_origin, ray_direction, cube_pos, cube.scale)
    if result[0] < 0:
        return None
    t, hx, hy, hz, nx, ny, nz = result
    return t, np.array([hx, hy, hz]), np.array([nx, ny, nz])

#-----------------------------------------------------------

def closest_intersection(ray_origin, ray_direction, geometry_cache):
    """Find closest intersection using JIT-compiled function with cached geometry."""
    result = _closest_intersection_jit(
        ray_origin, ray_direction,
        geometry_cache['sphere_positions'], geometry_cache['sphere_radii'], geometry_cache['sphere_mat_indices'],
        geometry_cache['plane_normals'], geometry_cache['plane_offsets'], geometry_cache['plane_mat_indices'],
        geometry_cache['cube_positions'], geometry_cache['cube_scales'], geometry_cache['cube_mat_indices']
    )

    if result[0] < 0:
        return None

    t, hx, hy, hz, nx, ny, nz, mat_idx, obj_type, obj_idx = result
    hit_point = np.array([hx, hy, hz])
    hit_normal = np.array([nx, ny, nz])

    return t, hit_point, hit_normal, int(mat_idx)

###########################   SHADOWING   ###############################

def build_light_plane(light_direction):
    # remember: light direction is from point to light
    # we can’t uniquely define a perpendicular vector to a given normal 
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


def soft_shadow(point_on_obj, light_src, geometry_cache, shadow_rays_num):
    # shadow_rays is given by scene settings
    shadow_rays_num = int(shadow_rays_num)
    total_shadow_rays = shadow_rays_num * shadow_rays_num

    light_vec = light_src.position - point_on_obj
    light_length = np.linalg.norm(light_vec)
    light_dir_norm = light_vec / light_length

    u, v = build_light_plane(light_dir_norm)

    # Vectorized generation of all shadow ray sample points
    # Create grid indices
    i_vals = np.arange(shadow_rays_num)
    j_vals = np.arange(shadow_rays_num)
    ii, jj = np.meshgrid(i_vals, j_vals, indexing='ij')
    ii = ii.flatten()
    jj = jj.flatten()

    # Generate all random offsets at once
    rand_i = np.random.rand(total_shadow_rays)
    rand_j = np.random.rand(total_shadow_rays)

    # Compute ru, rv for all rays
    ru = (ii + rand_i) / shadow_rays_num - 0.5
    rv = (jj + rand_j) / shadow_rays_num - 0.5

    # Compute all offset vectors: (ru * u + rv * v) * radius
    # u and v are (3,), ru and rv are (N,)
    offset_vectors = np.outer(ru, u) + np.outer(rv, v)  # (N, 3)
    offset_vectors *= light_src.radius

    # All light sample positions
    light_positions = light_src.position + offset_vectors  # (N, 3)

    # All shadow rays from point to light samples
    shadow_rays = light_positions - point_on_obj  # (N, 3)
    distances = np.linalg.norm(shadow_rays, axis=1)  # (N,)
    shadow_ray_dirs = shadow_rays / distances[:, np.newaxis]  # (N, 3)

    # Shadow ray origins (offset by epsilon)
    shadow_origins = point_on_obj + EPSILON * shadow_ray_dirs  # (N, 3)

    # Now test each ray (intersection tests are hard to vectorize due to early-exit)
    rays_reaching_light = 0
    for k in range(total_shadow_rays):
        hit = closest_intersection(shadow_origins[k], shadow_ray_dirs[k], geometry_cache)
        if hit is None:
            rays_reaching_light += 1
        else:
            t, hit_point, hit_normal, mat_idx = hit
            if t > distances[k]:
                rays_reaching_light += 1

    ratio = rays_reaching_light / total_shadow_rays

    # light always exists but the shadow reduces its intensity
    return (1 - light_src.shadow_intensity) + light_src.shadow_intensity * ratio


###########################   LIGHTING   ###############################

def compute_diffuse(Kd, Ip, normal, light_dir):
    # Idiff​=Kd​⋅Ip​⋅(N⋅L)
    # Kd = material.diffuse_color
    # Ip = light.color
    # N = normal
    # L = light_dir (normalized)

    # diffuse cant be negative
    N_dot_L = max(0.0, np.dot(normal, light_dir))

    return Kd * Ip * N_dot_L



#-----------------------------------------------------------


def compute_specular(Ks, Ip, shininess, normal, light_dir, view_dir):
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

    return Ks * Ip * (R_dot_V ** shininess)

#-----------------------------------------------------------

def compute_lighting(point_on_obj, normal, view_dir, material, lights, geometry_cache, scene_settings):

    # RGB color initialized to black
    color = np.zeros(3)

    # pre-convert material colors to numpy arrays (so no repeated conversion in inner loop)
    Kd = np.array(material.diffuse_color, dtype=float)
    Ks = np.array(material.specular_color, dtype=float)
    shininess = material.shininess

    for light in lights:
        # pre-convert light color to numpy array
        Ip = np.array(light.color, dtype=float)
        light_pos = np.array(light.position, dtype=float)

        # build light ray
        light_vec = light_pos - point_on_obj
        light_length = np.linalg.norm(light_vec)
        light_dir_norm = light_vec / light_length

        diffuse = compute_diffuse(Kd, Ip, normal, light_dir_norm)
        specular = compute_specular(Ks, Ip, shininess, normal, light_dir_norm, view_dir)
        specular *= light.specular_intensity
        soft_shadow_factor = soft_shadow(point_on_obj, light, geometry_cache, scene_settings.root_number_shadow_rays)

        color += soft_shadow_factor * (diffuse + specular)

    return np.clip(color, 0, 1)


###########################   REFLECTION   ###############################

def reflect(direction, normal):
    # calc return ray
    # R = V - 2(V⋅N)N
    return direction - 2 * np.dot(direction, normal) * normal

###########################   RAY TRACER   ###############################

def ray_tracer(camera, scene_settings, objects, image_width, image_height):
    # split scene objects and pre-cache geometry
    obj_lst, lights, materials, geometry_cache = organize_scene(objects)

    # render image
    image = render_scene(camera, scene_settings, geometry_cache, lights, materials, image_width, image_height)

    return image

#-----------------------------------------------------------

def rec_ray_tracer(ray_origin, ray_direction, depth, scene_settings, geometry_cache, lights, materials):

    # stopping condition
    # from now on, rays contribute no light
    if depth <= 0:
        return np.array(scene_settings.background_color, dtype=float)

    hit = closest_intersection(ray_origin, ray_direction, geometry_cache)

    # ray missed all objects, therefore return background color
    if hit is None:
        return np.array(scene_settings.background_color, dtype=float)

    t, hit_point, normal, mat_idx = hit
    material = materials[mat_idx - 1]

    # specular depends on the view direction
    # calc view direction (towards camera)
    # ray direction is from camera to point, so view direction is the opposite
    view_dir = -ray_direction
    view_dir_norm = view_dir / np.linalg.norm(view_dir)

    # calc lighting = diffuse + specular + soft shadow
    color = compute_lighting(hit_point, normal, view_dir_norm, material, lights, geometry_cache, scene_settings)

    # calc color returning from reflection
    # if the material has reflection color (one of the RGB is non zero)
    if np.any(material.reflection_color):
        reflect_dir = reflect(ray_direction, normal)
        reflect_dir_norm = reflect_dir / np.linalg.norm(reflect_dir)
        reflect_origin = hit_point + EPSILON * reflect_dir_norm

        reflected_color = rec_ray_tracer(
            reflect_origin,
            reflect_dir_norm,
            depth - 1,
            scene_settings,
            geometry_cache,
            lights,
            materials
        )

        Kr = np.array(material.reflection_color, dtype=float)
        # the returning color adds to the original color
        color += Kr * reflected_color


    # calc color returning from transparency
    if material.transparency > 0:
        trans_origin = hit_point + EPSILON * ray_direction

        transparent_color = rec_ray_tracer(
            trans_origin,
            ray_direction,
            depth - 1,
            scene_settings,
            geometry_cache,
            lights,
            materials
        )

        # the returning color blends with the original color
        color = (
            (1 - material.transparency) * color +
            material.transparency * transparent_color
        )

    return np.clip(color, 0, 1)


###########################   RENDER SCENE   ###############################

def _init_worker(cam_pos, screen_center, image_right_norm, image_up_norm,
                 screen_width, screen_height, image_width, image_height,
                 max_recursions, scene_settings, geometry_cache, lights, materials):
    """Initialize worker process with shared data."""
    _worker_data['cam_pos'] = cam_pos
    _worker_data['screen_center'] = screen_center
    _worker_data['image_right_norm'] = image_right_norm
    _worker_data['image_up_norm'] = image_up_norm
    _worker_data['screen_width'] = screen_width
    _worker_data['screen_height'] = screen_height
    _worker_data['image_width'] = image_width
    _worker_data['image_height'] = image_height
    _worker_data['max_recursions'] = max_recursions
    _worker_data['scene_settings'] = scene_settings
    _worker_data['geometry_cache'] = geometry_cache
    _worker_data['lights'] = lights
    _worker_data['materials'] = materials


def _render_row(y):
    """Render a single row of pixels. Called by worker processes."""
    d = _worker_data
    row = np.zeros((d['image_width'], 3))

    for x in range(d['image_width']):
        # normalized pixel coordinates in range [-0.5, 0.5]
        px = (x + 0.5) / d['image_width'] - 0.5
        py = (y + 0.5) / d['image_height'] - 0.5

        # pixel position on the screen
        pixel_pos = (
            d['screen_center'] +
            px * d['screen_width'] * d['image_right_norm'] +
            py * d['screen_height'] * d['image_up_norm']
        )

        # build ray
        ray_dir = pixel_pos - d['cam_pos']
        ray_dir_norm = ray_dir / np.linalg.norm(ray_dir)

        # trace ray
        color = rec_ray_tracer(
            d['cam_pos'],
            ray_dir_norm,
            d['max_recursions'],
            d['scene_settings'],
            d['geometry_cache'],
            d['lights'],
            d['materials']
        )

        row[x] = color * 255

    return y, row


def render_scene(camera, scene_settings, geometry_cache, lights, materials, image_width, image_height):
    # up - a vector from the center of the screen to the top of the screen
    # right - a vector from the center of the screen to the right of the screen

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

    # Prepare data for workers
    init_args = (
        cam_pos, screen_center, image_right_norm, image_up_norm,
        screen_width, screen_height, image_width, image_height,
        scene_settings.max_recursions, scene_settings, geometry_cache, lights, materials
    )

    # image array with RGB values
    image = np.zeros((image_height, image_width, 3))

    num_processes = cpu_count()
    print(f"Rendering with {num_processes} processes...")
    start_time = time.time()

    # Use multiprocessing pool to render rows in parallel
    with Pool(processes=num_processes, initializer=_init_worker, initargs=init_args) as pool:
        # Use imap_unordered for better progress tracking
        completed = 0
        for y, row in pool.imap_unordered(_render_row, range(image_height)):
            image[y] = row
            completed += 1
            elapsed = time.time() - start_time
            if completed > 0:
                eta = elapsed / completed * (image_height - completed)
                print(f"Row {completed}/{image_height} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s    ", end='\r')

    print()  # newline after progress
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


def save_image(image_array, output_path):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(output_path)

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    time_start = time.time()

    image_array = ray_tracer(
        camera,
        scene_settings,
        objects,
        args.width,
        args.height
    )

    time_end = time.time()
    print(f"Rendering time: {time_end - time_start:.2f} seconds")

    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
