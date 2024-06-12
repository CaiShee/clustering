# Pedestrian Trajectory Clustering Experiment

Welcome to this pedestrian trajectory clustering experiment project. This project aims to automatically discover and cluster pedestrian groups in video frames using clustering algorithms and to visualize the results. We have employed classic clustering algorithms such as K-means, DBSCAN, and hierarchical clustering, and evaluated the performance of the algorithms using internal metrics such as the Silhouette Coefficient, Calinski-Harabasz Index, and Davies-Bouldin Index.

## Project Overview

- **Experiment Objective**: Automatically discover and cluster pedestrian trajectories in videos and visualize the results.
- **Dataset**: students003 dataset.
- **Clustering Algorithms**: K-means, DBSCAN, Hierarchical Clustering (AGNES).
- **Performance Evaluation Metrics**: Silhouette Coefficient, CH Index, DB Index.

## Environment Requirements

- Python $\geq 3.8$
- NumPy
- SciPy
- sklearn
- matplotlib
- PyTorch

## Experimental Results

- K-means outperforms the other algorithms in the Silhouette Coefficient and CH Index.
- DBSCAN has the best performance in the DB Index.

