class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position # center of the sphere (numpy array)
        self.radius = radius
        self.material_index = material_index
