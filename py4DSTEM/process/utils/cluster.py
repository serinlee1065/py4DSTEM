import numpy as np
import matplotlib.pyplot as plt
import pymatgen
from scipy.ndimage import binary_erosion
from py4DSTEM.process.utils import tqdmnd
from scipy.ndimage import gaussian_filter


class Cluster:
    """
    Clustering 4D data

    """

    def __init__(
        self,
        datacube,
        r_space_mask,
    ):
        """
        Args:
            datacube (py4DSTEM.DataCube):         4D-STEM data
            r_space_mask (np.ndarray):            Mask in real space to apply background thresholding on the similarity array.

        """
        self.datacube = datacube
        self.r_space_mask = r_space_mask
        self.similarity = None
        self.similarity_raw = None

    def bg_thresholding(self, r_space_mask,):
        self.r_space_mask = np.asarray(r_space_mask)

        # if similarity is already computed, apply the thresholding
        if self.similarity_raw is not None:
            self.similarity = self._apply_bg_mask(self.similarity_raw)

    def _apply_bg_mask(self, similarity):
        if self.r_space_mask is None:
            return similarity
        return similarity * self.r_space_mask[..., None]


    def find_similarity(
        self,
        q_space_mask=None,  
        smooth_sigma = 0,
        return_similarity = False
    ):
        
        """
        Args:
            q_space_mask : annular boolean q_space_mask to apply on the diffraction patterns
            smooth_sigma : sigma for Gaussian smoothing of the diffraction patterns before calculating similarity
            return_similarity : if True, return the similarity array
        """
        if self.r_space_mask is None:
            self.set_mask(r_space_mask)
        
        # List of neighbors to search
        # (-1,-1) will be equivalent to (1,1)
        self.dxy = np.array(
            (
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (1, 1),
                (1, 0),
                (1, -1),
                (0, 1),
            )
        )

        # initialize the self.similarity array
        self.similarity = -1 * np.ones(
            (self.datacube.shape[0], self.datacube.shape[1], self.dxy.shape[0])
        )

        # Loop over probe positions
        for rx, ry in tqdmnd(
            range(self.datacube.shape[0]),
            range(self.datacube.shape[1]),
        ):
            diff_ref = self.datacube[rx, ry].copy().astype('float')
            diff_ref -= diff_ref.mean()

            if smooth_sigma > 0:
                diff_ref = gaussian_filter(diff_ref,smooth_sigma)
            
            if q_space_mask is not None:
                diff_ref = diff_ref[q_space_mask]

            norm_diff_ref = np.sqrt(np.sum(diff_ref * diff_ref))
            # diff_ref_mean = np.mean(diff_ref)

            # loop over neighbors
            for ind in range(self.dxy.shape[0]):
                x_ind = rx + self.dxy[ind, 0]
                y_ind = ry + self.dxy[ind, 1]
                if (
                    x_ind >= 0
                    and y_ind >= 0
                    and x_ind < self.datacube.shape[0]
                    and y_ind < self.datacube.shape[1]
                ):
                    diff = self.datacube[x_ind, y_ind].copy().astype('float')
                    diff -= diff.mean()

                    if smooth_sigma > 0:
                        diff = gaussian_filter(diff,smooth_sigma)
                    
                    if q_space_mask is not None:
                        diff = diff[q_space_mask]
        
                    # image self.similarity with normalized cosine correlation
                    self.similarity[rx, ry, ind] = (
                        np.sum(diff * diff_ref)
                        / np.sqrt(np.sum(diff * diff))
                        / norm_diff_ref
                    )

                    
        self.similarity_raw = self.similarity.copy()
        self.similarity = self._apply_bg_mask(self.similarity)

        if return_similarity:
            return self.similarity
        
    # Create a function to map cluster index to color
    def get_color(self, cluster_index):
        colors = [
            "slategray",
            "lightcoral",
            "gold",
            "darkorange",
            "yellowgreen",
            "lightseagreen",
            "cornflowerblue",
            "royalblue",
            "lightsteelblue",
            "darkseagreen",
        ]
        return colors[(cluster_index - 1) % len(colors)]

    # Find the pixel with the highest self.similarity and start the clustering from there
    def indexing_clusters_all(
        self,
        threshold,
    ):
        """
        Args:
            threshold : threshold for similarity to consider two pixels as part of the same cluster
        """
        
        sim_averaged = np.mean(self.similarity, axis=2)

        # Assigning the background as 'counted' 
        sim_averaged[~self.r_space_mask] = -1.0 

        # color the pixels with the cluster index
        self.cluster_map = -1 * np.ones(
            (sim_averaged.shape[0], sim_averaged.shape[1]), dtype=np.float64
        )
        self.cluster_map_rgb = np.zeros(
            (sim_averaged.shape[0], sim_averaged.shape[1], 4), dtype=np.float64
        )
       
        self.cluster_map_rgb[..., 3] = 1.0   #start as opaque black

        # store arrays of cluster_indices in a list
        self.cluster_list = []

        # incides of pixel in a cluster
        cluster_indices = np.empty((0, 2))

        # Loop over pixels until no new pixel is found (sim_averaged is set to -1 if it is alreaddy serached for NN)
        cluster_count_ind = 0

        while np.any(sim_averaged != -1):

            # finding the pixel that has the highest self.similarity among the pixel that hasn't been clustered yet
            # this will be the 'starting pixel' of a new cluster
            rx0, ry0 = np.unravel_index(sim_averaged.argmax(), sim_averaged.shape)
        
            # Guarding to check if the seed is background
            if self.r_space_mask is not None and not self.r_space_mask[rx0, ry0]:
                sim_averaged[rx0, ry0] = -1  # mark processed so we don't pick it again
                continue


            cluster_indices = np.empty((0, 2))
            cluster_indices = (np.append(cluster_indices, [[rx0, ry0]], axis=0)).astype(
                np.int32
            )

            self.cluster_map[rx0, ry0] = cluster_count_ind

            color = self.get_color(cluster_count_ind + 1)
            self.cluster_map_rgb[rx0, ry0] = plt.cm.colors.to_rgba(color)

            # Clustering: one cluster per while loop(until it breaks)
            # Marching algorithm: find a new position and search the nearest neighbor

            while True:
                counting_added_pixel = 0

                for rx0, ry0 in cluster_indices:

                    if sim_averaged[rx0, ry0] != -1:

                        # counter to check if pixel in the cluster are checked for NN
                        counting_added_pixel += 1

                        # set to -1 since now its NN will be checked
                        sim_averaged[rx0, ry0] = -1

                        for ind in range(self.dxy.shape[0]):
                            x_ind = rx0 + self.dxy[ind, 0]
                            y_ind = ry0 + self.dxy[ind, 1]

                            if x_ind > 1 and \
                                y_ind > 1 and \
                                x_ind < self.similarity.shape[0] - 2 and \
                                y_ind < self.similarity.shape[1] - 2:

                                r_ok = True if self.r_space_mask is None else bool(self.r_space_mask[x_ind, y_ind])

                                # add if the neighbor is similar, but don't add if the neighbor is already in a cluster
                                if self.similarity[rx0, ry0, ind] >= threshold \
                                    and self.cluster_map[x_ind, y_ind] == -1 and r_ok:

                                    
                                    cluster_indices = np.append(
                                        cluster_indices, [[x_ind, y_ind]], axis=0
                                    )

                                    self.cluster_map[x_ind, y_ind] = cluster_count_ind
                                

                                    # self.cluster_map[x_ind, y_ind] = cluster_count_ind+1
                                    color = self.get_color(cluster_count_ind + 1)
                                    self.cluster_map_rgb[x_ind, y_ind] = plt.cm.colors.to_rgba(
                                        color
                                    )

                # if no new pixel is checked for NN then break
                if counting_added_pixel == 0:
                    break

            # single pixel cluster
            # if cluster_indices.shape[0] == 1:
            #     self.cluster_map_rgb[cluster_indices[0, 0], cluster_indices[0, 1]] = [
            #         0,
            #         0,
            #         0,
            #         1,
            #     ]

            self.cluster_list.append(cluster_indices)
            cluster_count_ind += 1

        # return cluster_count_ind, self.cluster_list, self.cluster_map, self.cluster_map_rgb

    def create_cluster_cube(
        self,
        min_cluster_size,
        return_cluster_datacube=False,
    ):

        self.filtered_cluster_list = [
            arr for arr in self.cluster_list if arr.shape[0] >= min_cluster_size
        ]

        # datacube [i,j,k,l] where i is the index of the cluster, and j is a place holder, and k,l are the average diffraction pattern of the
        self.cluster_cube = np.empty(
            [
                len(self.filtered_cluster_list),
                1,
                self.datacube.shape[2],
                self.datacube.shape[3],
            ]
        )

        for i in tqdmnd(range(len(self.filtered_cluster_list))):
            self.cluster_cube[i, 0] = self.datacube[
                np.array(self.filtered_cluster_list[i])[:, 0],
                np.array(self.filtered_cluster_list[i])[:, 1],
            ].mean(axis=0)

        if return_cluster_datacube:
            return self.cluster_cube, self.filtered_cluster_list
