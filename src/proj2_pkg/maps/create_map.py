#!/usr/bin/env python

"""
Starter code for EECS C106B Spring 2020 Project 2.
Author: Tiffany Cappellari
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image

class Obstacle():
	def __init__(self, center_x, center_y, radius):
		self.center = (center_x, center_y)
		self.radius = radius

def create_grid(obstacle_list, width, height, resolution=0.01, terrains=[]):
	"""Creates a grid representing the map with obstacles
	in obstacle_list.

	Args:
		obstacle_list: 'list' of Obstacle objects. Assumes center
		and radius are given in meters.
		width: width of the environment (expanse in x direction) in meters.
		height: height of the environment (expanse in y direction) in meters.
		resolution: meters per pixel.
	"""
	def m_to_pix(*nums):
		return [int(x / resolution) for x in nums]

	m = np.ones(m_to_pix(height, width) + [3])
	# for obstacle in obstacle_list:
	# 	x_c, y_c = m_to_pix(*obstacle.center)
	# 	r, = m_to_pix(obstacle.radius)
	# 	for x in range(x_c - r, x_c + r):
	# 		for y in range(y_c - r, y_c + r):
	# 			if ((x - x_c)**2 + (y - y_c)**2) <= r**2:
	# 				m[y][x] = 0

	for terrain in terrains:
		x1, x2, y1, y2 = m_to_pix(*(terrain[0]))
		d, k = (terrain[1][0], terrain[2][0])
		for x in range(x1, x2):
			for y in range(y1, y2):
				m[y][x] = [1, d, k]

	m = m[::-1]
	return m

def create_png(m, name):
	# scipy.misc.imsave(name + '.png', m)
	im = Image.fromarray((m * 255).astype(np.uint8))
	im.save(name + '.png')

def create_yaml(png, name, resolution=0.01):
	f = open('{}.yaml'.format(name), 'w')
	f.write('image: {}\n'.format(png))
	f.write('resolution: ' + str(resolution) + '\n')
	f.write('origin: [0.0, 0.0, 0.0]\n')
	f.write('occupied_thresh: 0.6\n')
	f.write('free_thresh: 0.3\n')
	f.write('negate: 0\n')

def create_map(obstacle_list, width, height, name, resolution=0.01, terrains=[]):
	m = create_grid(obstacle_list, width, height, resolution, terrains=terrains)
	create_png(m, name)
	create_yaml("{}.png".format(name), name, resolution=resolution)

def make_map1():
	obstacle1 = Obstacle(6, 3.5, 1.5)
	obstacle2 = Obstacle(3.5, 6.5, 1)
	create_map([obstacle1, obstacle2], 10, 10, "map1")

def make_map2():
	obstacle1 = Obstacle(2, 5, 1)
	obstacle2 = Obstacle(5, 5, 1)
	obstacle3 = Obstacle(8, 5, 1)
	obstacle4 = Obstacle(8, 2, 1)
	obstacle5 = Obstacle(2, 8, 1)
	create_map([obstacle1, obstacle2, obstacle3, obstacle4, obstacle5], 10, 10, "map2")

def make_empty_map():
	create_map([], 5, 5, "empty")

def make_terrain_map1():
	create_map([], 10, 4, "terrain_noavoid", terrains=[[[3, 6], [0, 4], 0.1, 1]], resolution=0.01)
	pass

def make_two_terrains():
	create_map([], 15, 15, "two_terrains", terrains=[[[2, 8, 0, 15], [0.5, 0.3], [0.5, 0.3]], [[2, 8, 0, 15], [0.5, 0.1], [0.5, 0.1]]])

def make_terrain_map2():
	x_len, y_len = 10, 10
	terrains = []
	terrains_launch = []
	res=1
	for i in range(res*x_len):
		for j in range(res*y_len):
			d = np.random.random()*0.8 + 0.2
			terrains.append([[i, i+1],[j,j+1], [d, 0], [1,0]])
			terrains_launch.append([[i, i+1,j,j+1], [d,0],[1,0]])
	create_map([], x_len, y_len, "terrains_random", terrains=terrains_launch, resolution=0.01)
	file = open("terrain_randmap.txt", "w")
	file.write(str(terrains_launch))
	file.close()

if __name__ == '__main__':
	# make_map1()
	# make_map2()
	# make_empty_map()
	# make_terrain_map()
	# make_two_terrains()
	make_terrain_map2()
