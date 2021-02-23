#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:07:25 2021
@author: alex

Implements a small model inspired by the book Guns, Germs and Steel by Jared
Diamond that describes why different peoples developed differently on a large
time scale (roughly the last 13000 years). Diamond only relies on environmental
arguments.
"""

import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from tqdm import tqdm

class GunsGermsSteel():
    """Class for modelling and plotting."""
    def __init__(self, map_path, spread_rate=1,
                 dev_rate=1, axis_ratio=0.1):
        """Reads map and sets up basic variables."""
        # save input
        self.map_name = map_path[:-4]
        self.geo_map = np.loadtxt(map_path).astype(int)
        self.spread_rate = spread_rate
        self.dev_rate = dev_rate
        self.axis_ratio = axis_ratio
        # additional parameters
        self.height = len(self.geo_map)
        self.width = len(self.geo_map[0])
        self.dev_map = np.zeros((self.height, self.width))
        self.inhabitable_coord = self.get_inhabitable_coordinates()
        self.n_cells = len(self.inhabitable_coord)

    def reset(self):
        """Resets all variables, so simulations can start from scratch."""
        self.dev_map = np.zeros((self.height, self.width))

    def get_inhabitable_coordinates(self):
        """Iterates through the whole map to get all coordinates (i, j) that
        are on inhabitable cells. Returns a list that can be used to iterate
        through all inhabitablecells or get random land cells."""
        inhabitable_coord = []
        for i in range(self.height):
            for j in range(self.width):
                if self.geo_map[i, j] == 0:
                    inhabitable_coord.append((i, j))
        return inhabitable_coord

    def get_dev_list(self):
        """Returns all development values on the dev_map."""
        devs = []
        for i, j in self.inhabitable_coord:
            devs.append(self.dev_map[i, j])
        return devs

    def get_dev_range(self, devs):
        """Calculates the range of all development values."""
        dev_min = min(devs)
        dev_max = max(devs)
        return dev_max - dev_min

    def get_shannon_entropy(self, devs):
        """Calculates shannon entropy with n_cells bins."""
        hist, _ = np.histogram(devs, bins=len(self.inhabitable_coord))
        p = hist / np.sum(hist)
        p = np.ma.masked_where(p <= 0, p)
        return -np.sum(p*np.log(p))

    def geo_color_map(self):
        """Returns colormap for geographic background. Water (white), land
        (grey), uninhabitable landcells (black)."""
        cmaplist = [(0.5, 0.5, 0.5), (1, 1, 1), (0, 0, 0)]
        boundaries = [-0.5, 0.5, 1.5, 2.5]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'custom_cmap', cmaplist, N=len(cmaplist))
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        return cmap, norm

    def show_dev_map(self, save_path=None):
        """Shows  or saves development map. Can be used for interactive mode
        or for saving mode."""
        plt.clf()
        # development map
        masked_dev_map = np.ma.masked_where(self.geo_map != 0, self.dev_map)
        geo_cmap, geo_norm = self.geo_color_map()
        plt.imshow(self.geo_map, cmap=geo_cmap, norm=geo_norm)
        plt.imshow(masked_dev_map, cmap='brg')
        plt.colorbar()
        # save or show
        plt.tight_layout()
        if save_path is not None:
            plt.ioff()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.pause(0.0001)

    def get_random_cell(self):
        """Gets indexes of a random cell that has civilization."""
        rand_index = np.random.randint(self.n_cells)
        return self.inhabitable_coord[rand_index]

    def get_neighbor(self, i, j):
        """Finds a possible neighbor that is inhabtiable from the 8 nearest
        neighbors.."""
        i_neigh = i + np.random.choice([-1, 0, 1])
        j_neigh = j + np.random.choice([-1, 0, 1])
        while (self.geo_map[i_neigh, j_neigh] != 0 \
               or (i == i_neigh and j == j_neigh)):
            i_neigh = i + np.random.choice([-1, 0, 1])
            j_neigh = j + np.random.choice([-1, 0, 1])
        return i_neigh, j_neigh

    def should_spread(self, i, i_neigh):
        """Returns true if development should spread to neighbor cell.
        Accounts for different axes."""
        spread = False
        rand_num = np.random.random()
        if i == i_neigh:
            spread = True
        elif rand_num < self.axis_ratio: # i != i_neigh, so North-South
            spread = True
        return spread

    def spread(self):
        """Spreads development for a random cell that is civilized."""
        i, j = self.get_random_cell()
        i_neigh, j_neigh = self.get_neighbor(i, j)
        if (self.dev_map[i, j] > self.dev_map[i_neigh, j_neigh]
                and self.should_spread(i, i_neigh)):
            self.dev_map[i_neigh, j_neigh] = self.dev_map[i, j]

    def develop(self):
        """Increases development for a random civilized cell."""
        i, j = self.get_random_cell()
        self.dev_map[i, j] += 1

    def time_step(self):
        """Normalizes model steps on number of inhabitable cells."""
        for _ in range(int(self.n_cells * self.spread_rate)):
            self.spread()
        for _ in range(int(self.n_cells * self.dev_rate)):
            self.develop()

    def simulate(self, steps, current_folder=None, save_interval=50, seed=None):
        """Simulating the model for a given amount of steps."""
        ranges = []
        shannon = []
        parameter_string = ('seed' + str(seed)  + 'ax'
                            + str(self.axis_ratio))
        for i in tqdm(range(steps)):
            self.time_step()
            devs = self.get_dev_list()
            ranges.append(self.get_dev_range(devs))
            shannon.append(self.get_shannon_entropy(devs))
            if (current_folder is not None
                    and ((i + 1)%save_interval == 0 or i == 4)):
                save_path = (current_folder + parameter_string +
                             '_it' + str(i + 1) + '.png')
                self.show_dev_map(save_path=save_path)
            elif current_folder is None:
                self.show_dev_map()
        if current_folder is not None:
            plt.clf()
            np.savetxt((current_folder + parameter_string + 'ranges.txt')
                       , ranges)
            plt.plot(ranges)
            plt.xlabel('iteration')
            plt.ylabel('range')
            plt.savefig((current_folder + parameter_string + 'ranges.png'))
            plt.clf()
            np.savetxt((current_folder + parameter_string + 'shannon.txt')
                       , shannon)
            plt.plot(shannon)
            plt.xlabel('iteration')
            plt.ylabel('shannon entropy')
            plt.savefig((current_folder + parameter_string + 'shannon.png'))
            plt.clf()

    def interactive(self, steps=10000, seed=0):
        """Simulates on time for a given random seeds and displays every
        iteration."""
        np.random.seed(seed)
        self.simulate(steps)

    def multi_sim(self, steps=5000, seed_start=0, seed_end=9,
                  save_folder='simulations', folder_ext='',
                  save_interval=1000):
        """Simulates multiple times for different random seeds and saves data
        """
        for seed in range(seed_start, seed_end + 1):
            np.random.seed(seed)
            self.reset()
            current_folder = (save_folder + '/' + self.map_name + folder_ext
                              + '/' + str(seed) + '/')
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)
            self.simulate(steps, current_folder=current_folder,
                          save_interval=save_interval, seed=seed)

def main():
    """Main function that is called when the script is executed."""
    #GunsGermsSteel('mollweide144x85.txt').interactive()
    GunsGermsSteel('mollweide144x85.txt', axis_ratio=0.1).multi_sim(folder_ext='_ax0.1')
    #GunsGermsSteel('mollweide144x85.txt', axis_ratio=0.3).multi_sim(folder_ext='_ax0.2')
    #GunsGermsSteel('mollweide144x85.txt', axis_ratio=1).multi_sim(folder_ext='_ax1')
    #GunsGermsSteel('mollweide144x85only_water_and_land.txt', axis_ratio=0.1).multi_sim(folder_ext='_ax0.1')

if __name__ == '__main__':
    main()
