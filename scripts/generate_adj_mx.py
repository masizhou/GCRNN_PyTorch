from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
import argparse


def get_adjacency_matrix(distance_df, station_ids, normalized_k=0.1):
    """
    generate adjacency matrix
    :param distance_df: data frame with three columns: [from, to, distance].
    :param station_ids: list of station ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_stations = len(station_ids)
    dist_mx = np.zeros((num_stations, num_stations), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds station id to index map.
    station_id_to_ind = {}
    for i, station_id in enumerate(station_ids):
        station_id_to_ind[station_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in station_id_to_ind or row[1] not in station_id_to_ind:
            continue
        dist_mx[station_id_to_ind[row[0]], station_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return station_ids, station_id_to_ind, adj_mx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--station_ids_filename', type=str, default='data/Heilongjiang_province/station_graph/heilongjiang_graph_station_ids',
                        help='File containing station ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='data/Heilongjiang_province/station_graph/distances_heilongjiang.csv',
                        help='CSV file containing station distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='data/Heilongjiang_province/station_graph/heilongjiang_adj_mat.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    with open(args.station_ids_filename) as f:
        station_ids = f.read().strip().split('\n') # list

    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
    _, station_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, station_ids)
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([station_ids, station_id_to_ind, adj_mx], f, protocol=2)