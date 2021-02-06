#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:07:25 2021
@author: alex
Implements a small model inspired by the book Guns, Germs and Steel by Jared
Diamond that describes why different peoples developed differently on a large
time scale (roughly the last 13000 years). Diamond only relies on environmental
arguments. This model focuses on the arguments of continent size and major
continental axes.
Abbreviations:
geo: geographical
pol: political
dev: development
civ: civilization
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from tqdm import tqdm

class GunsGermsSteel():
    """Class for modelling and plotting."""
    def __init__(self, map_path):
        """Reads map and sets up basic variables needed for the model and
        plotting. The map should be a .txt file with zeros (water) and ones
        (land). Map should have water (zeros) at all borders. To save time,
        model does not have periodic boundaries. No small unconnected  islands
        should be on the map, or the development map will show mostly green.
        geo_map will be the plain read in map. pol_map will show the different
        peoples on a categorical color map. On the pol_map 0 is water, 1
        uncivilized hunter and gatherer. Every number bigger than 1 will be
        assigned to another people that has launched civilization. civ_index
        will be used to count the index number for a civilizaiton. dev_map
        will show the current development of a cell (pixel). For hunter and
        gatherers (pol_map == 1) dev_map will always be 0.
        The cmaplist will give every people a random color. Water will be white
        and hunter and gatherers will be always grey. boundaries corresponds
        to the cmaplist."""
        # maps and its variables
        self.geo_map = np.loadtxt(map_path).astype(int)
        self.height = len(self.geo_map)
        self.width = len(self.geo_map[0])
        self.pol_map = np.copy(self.geo_map)
        self.civ_index = 1 # starting with 1 because 2 will be the next valid index
        self.dev_map = np.zeros((self.height, self.width))
        self.liv_coord = self.get_livable_coordinates()
        self.n_civ_timeseries = []
        # colormap variables
        self.cmaplist = [(1, 1, 1), (0.5, 0.5, 0.5)]
        self.boundaries = [-0.5, 0.5, 1.5]

    def get_livable_coordinates(self):
        """Iterates through the whole map to get all coordinates (i, j) that
        are on livable land. Returns a list that can be used to iterate through
        all livable land cells or get random land cells."""
        liv_coord = []
        for i in range(self.height):
            for j in range(self.width):
                if self.geo_map[i, j] == 1:
                    liv_coord.append((i, j))
        return liv_coord

    def get_cells_per_people(self):
        """Counts cells for every civilization."""
        cells_per_civ = np.zeros(len(self.cmaplist))
        for i, j in self.liv_coord:
            civ_index = self.pol_map[i, j]
            cells_per_civ[civ_index] += 1
        return cells_per_civ

    def update_pol_indexes(self):
        """Kicks indexes for peoples that got extinct, so colors
        (and its boundaries) can be updated.
        """
        cells_per_civ = self.get_cells_per_people()
        # update cmaplist and boundaries
        new_cmaplist = [(1, 1, 1), (0.5, 0.5, 0.5)]
        old_to_new_index = [0, 1]
        new_index = 1
        # first two are water and hunter gatherer
        for index, n_cells in enumerate(cells_per_civ[2:]):
            if n_cells > 0:
                new_cmaplist.append(self.cmaplist[index + 2])
                new_index += 1
                old_to_new_index.append(new_index)
            else:
                old_to_new_index.append('x')
                self.civ_index -= 1
        self.cmaplist = new_cmaplist
        self.boundaries = self.boundaries[:(len(self.cmaplist) + 1)]
        # update polititcal map and number of civiliations
        for i, j in self.liv_coord:
            old_civ_index = self.pol_map[i, j]
            self.pol_map[i, j] = old_to_new_index[old_civ_index]

    def get_current_colormap(self):
        """Creates the current color map for plotting the political map."""
        self.update_pol_indexes()
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'custom_cmap', self.cmaplist, N=len(self.cmaplist))
        norm = mpl.colors.BoundaryNorm(self.boundaries, cmap.N, clip=True)
        return cmap, norm

    def get_dev_list(self):
        """Returns all development values on the dev_map."""
        devs = []
        for i, j in self.liv_coord:
            devs.append(self.dev_map[i, j])
        return devs

    def show_pol_and_dev_map(self, save_path=None):
        """Shows political and development map simultaneously."""
        plt.clf()
        # political map
        plt.subplot(2, 2, 1)
        cmap, norm = self.get_current_colormap()
        plt.imshow(self.pol_map, cmap=cmap, norm=norm)
        plt.title('Political Map')
        cbar = plt.colorbar()
        cbar.set_ticks([])
        # development map
        plt.subplot(2, 2, 2)
        masked_dev_map = np.ma.masked_where(self.geo_map == 0, self.dev_map)
        plt.imshow(masked_dev_map, cmap='RdYlGn')
        plt.title('Relative Development Map')
        plt.colorbar()
        # time and number of civilizations
        plt.subplot(2, 2, 3)
        plt.title('Number of Civilizations')
        plt.plot(self.n_civ_timeseries)
        plt.xlabel('Iteration')
        # dev histogram
        plt.subplot(2, 2, 4)
        plt.title('Development Histogram')
        plt.hist(self.get_dev_list(), bins=30, histtype='bar')
        plt.xlabel('Development')
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
        """Gets indexes of a random cell that is on land."""
        rand_index = np.random.randint(len(self.liv_coord))
        return self.liv_coord[rand_index]

    def get_neighbor(self, i, j):
        """Finds a possible neighbor that is on land from the 8 nearest
        neighbors. Could be of the same people."""
        i_neigh = i + np.random.choice([-1, 0, 1])
        j_neigh = j + np.random.choice([-1, 0, 1])
        while (self.geo_map[i_neigh, j_neigh] == 0 \
               or (i == i_neigh and j == j_neigh)):
            i_neigh = i + np.random.choice([-1, 0, 1])
            j_neigh = j + np.random.choice([-1, 0, 1])
        return i_neigh, j_neigh

    def maybe_spawn_civ(self):
        """Spawns only one new civilization with some probability, somewhere on
        the map. Can also be a splitting off from an existing civilization."""
        self.n_civ_timeseries.append(self.civ_index - 1)
        rand_num = np.random.random()
        if rand_num < self.p_spawn:
            i, j = self.get_random_cell()
            self.civ_index += 1
            self.pol_map[i, j] = self.civ_index
            self.dev_map[i, j] += 8 # headstart, so a split off can survive
            # update colormap to give new civilization a color
            new_color = (np.random.random(), np.random.random(),
                         np.random.random())
            self.cmaplist.append(new_color)
            self.boundaries.append(len(self.cmaplist) - 0.5)

    def maybe_engulf(self, i, j, i_neigh, j_neigh):
        """If one cell attacks, decides if it engulfs the cell."""
        if self.pol_map[i, j] != self.pol_map[i_neigh, j_neigh]:
            if self.dev_map[i, j] > self.dev_map[i_neigh, j_neigh]:
                self.pol_map[i_neigh, j_neigh] = self.pol_map[i, j]

    def maybe_attack(self):
        """Every cell might atack a surrounding cell."""
        for i, j in self.liv_coord:
            rand_num = np.random.random()
            if rand_num < self.p_attack and self.geo_map[i, j] == 1:
                i_neigh, j_neigh = self.get_neighbor(i, j)
                self.maybe_engulf(i, j, i_neigh, j_neigh)

    def maybe_develop(self):
        """Gives every cell an equal probability to develop."""
        for i, j in self.liv_coord:
            if self.pol_map[i, j] > 1:
                rand_num = np.random.random()
                if rand_num < self.p_develop:
                    self.dev_map[i, j] += 1

    def spread_decision(self, i, j, i_neigh, j_neigh):
        """Returns true if development should spread to neighbor cell."""
        spread = False
        civ = self.pol_map[i, j]
        neigh_civ = self.pol_map[i_neigh, j_neigh]
        if civ > 1 and neigh_civ > 1:
            rand_num = np.random.random()
            if civ == neigh_civ:
                if i == i_neigh and rand_num < self.p_spread_hor:
                    spread = True
                elif i != i_neigh and rand_num > (1 -self.p_spread_ver):
                    spread = True
            else:
                if i == i_neigh and rand_num < self.p_spread_border_hor:
                    spread = True
                elif i != i_neigh and rand_num > (1 -self.p_spread_border_ver):
                    spread = True
        return spread

    def spread_dev(self):
        """Spreads development. Random cells are chosen."""
        for _ in self.liv_coord:
            i, j = self.get_random_cell()
            i_neigh, j_neigh = self.get_neighbor(i, j)
            if (self.dev_map[i, j] > self.dev_map[i_neigh, j_neigh]
                    and self.spread_decision(i, j, i_neigh, j_neigh)):
                self.dev_map[i_neigh, j_neigh] = self.dev_map[i, j]

    def time_step(self):
        """Adds all parts of the model together"""
        #for _ in range(self.height * self.width):
        self.maybe_spawn_civ() # only one cell per time step
        self.maybe_attack() # every cell
        self.maybe_develop() # every cell
        self.spread_dev() # every cell (on average)

    def simulate(self, steps, p=1, bor_ratio=0.1,
                 current_folder=None, save_interval=50):
        """Simulating the model for a given amount of steps."""
        # probabilities
        spread_ratio_hor_ver = 0.1
        spread_ratio_border = bor_ratio
        self.p_spawn = p
        self.p_attack = p
        self.p_spread_hor = p
        self.p_spread_ver = self.p_spread_hor * spread_ratio_hor_ver
        self.p_spread_border_hor = self.p_spread_hor * spread_ratio_border
        self.p_spread_border_ver = self.p_spread_ver * spread_ratio_border
        self.p_develop = p
        # simulation
        for i in tqdm(range(steps)):
            self.time_step()
            if (current_folder is not None
                    and (i%save_interval == 0 or i == 10)):
                save_path = current_folder + str(i) + '.png'
                self.show_pol_and_dev_map(save_path=save_path)
            elif current_folder is None:
                self.show_pol_and_dev_map()

def interactive_simulation(map_path, steps, seed=0, p=1, bor_ratio=0.1):
    """Simulates on time for a given random seeds and displays every iteration.
    """
    np.random.seed(seed)
    ggs = GunsGermsSteel(map_path)
    ggs.simulate(steps, p=p, bor_ratio=bor_ratio)

def multiple_simulatations(map_path, steps, seed_start, seed_end, p=1,
                           bor_ratio=0.1,
                           simulation_folder='multiple_simulations',
                           save_interval=50):
    """Simulates multiple times for different random seeds and saves every 100
    iterations."""
    for seed in range(seed_start, seed_end + 1):
        np.random.seed(seed)
        ggs = GunsGermsSteel(map_path)
        current_folder = (simulation_folder + '/' + map_path[:-4]
                          + '_bor_ratio' + str(bor_ratio)
                          + '/' + str(seed) + '/')
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
        ggs.simulate(steps, p=p, bor_ratio=bor_ratio,
                     current_folder=current_folder,
                     save_interval=save_interval)

def main():
    """Main function that is called when the script is executed."""
    #interactive_simulation('mollweide144x85.txt', 5000, seed=0)
    #multiple_simulatations('mollweide144x85rot.txt', 5000, 0, 2)
    #multiple_simulatations('mollweide144x85.txt', 5000, 0, 2)
    multiple_simulatations('mollweide144x85rot.txt', 5000, 0, 2,
                           bor_ratio=0.001)
    multiple_simulatations('mollweide144x85.txt', 5000, 0, 2,
                           bor_ratio=0.001)
    multiple_simulatations('mollweide144x85rot.txt', 5000, 0, 2,
                           bor_ratio=0.0001)
    multiple_simulatations('mollweide144x85.txt', 5000, 0, 2,
                           bor_ratio=0.0001)

if __name__ == '__main__':
    main()
