from gimpfu import pdb, main, register, PF_STRING
from gimpenums import ORIENTATION_HORIZONTAL
import random
import os

def lighting_effect(file):
    image = pdb.gimp_file_load(file, file)
    drawable = pdb.gimp_image_get_active_layer(image)
    lighting_position_x = random.uniform(-1, 1)
    lighting_position_y = random.uniform(-1, 1)
    lighting_position_z = random.uniform(0.1,0.5)
    lighting_direction_x = random.uniform(-1, 1)
    lighting_direction_y = random.uniform(-1, 1)
    lighting_direction_z = random.uniform(-1, 1)
    ambient_intensity = random.uniform(0.4, 0.9) 
    diffuse_intensity = random.uniform(0, 0.5)
    diffuse_reflectivity = random.uniform(0, 0.5)
    specular_reflectivity = random.uniform(0, 0.5)
    highlight = random.uniform(0, 1)
    pdb.plug_in_lighting(image, drawable, None, None, False, False, 0, 0, (255,255,255), lighting_position_x, lighting_position_y, lighting_position_z, lighting_direction_x, lighting_direction_y, lighting_direction_z, ambient_intensity, diffuse_intensity, diffuse_reflectivity, specular_reflectivity, highlight, False, False, False)
    pdb.gimp_file_save(image, drawable, file, file)
    pdb.gimp_image_delete(image)

all_files = os.listdir('Desktop/UVA/Scriptie/images/lp')
for file in all_files:
    lighting_effect('Desktop/UVA/Scriptie/images/lp/' + file)

