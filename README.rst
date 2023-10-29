Routing Library
===============

A simple routing library implementing algorithms available in literature

Installation
============

With Python and Pip installed, do:

.. code:: sh


      pip install routing-lib
      

Functionalities
===============

There are 3 modules which can be imported as it follows:

.. code:: sh


      from routing_lib.routing_algorithms import *
      from routing_lib.routing_measures import *
      from routing_lib.routing_utils import *
      

Algorithms
----------

The ``routing_algorithms`` module provides various routing algorithms:

-  ``apply_penalization``: Applies penalization to edges in a graph
   based on a specific weight update criteria.
-  ``path_penalization``: This function applies penalization to edge
   weights specifically for shortest paths in a graph. It uses a
   specified function to update edge weights iteratively.
-  ``graph_randomization``: Randomizes weights for all edges in a graph
   to find k paths with updated costs. It extract the penalization
   values froma a random distribution function.
-  ``path_randomization``: Performs randomization on edge weights
   specifically for shortest paths in a graph, like before extracting
   penalization values from a random distribution.
-  ``duarouter``: Applies edge weight adjustments using a specified
   disturbance factor.
-  ``k_disjointed``: Penalizes edges to infinity to get to the results
   of quasi-disjointed paths.
-  ``no_randomization``: Applies no randomization to edge weights.
-  ``k_mdnsp``: This function applies a specified function to update
   edge weights for finding k paths with cost under a certain threshold
   while ensuring the maximum possible level of dissimilarity.
-  ``next_ssvp_by_length``: This function generates the next simple
   single-via path based on edge lengths in a graph.
-  ``kspml``: This function provides k-shortest paths with minimum
   collective length, by using the SSVP.
-  ``kspmo``: This function is similar to ‘kspml’ but employs a
   different dissimilarity metric to evaluate the similarity between
   paths.
-  ``plateau_algorithm``: This function runs the plateau-based algorithm
   to find k plateaus and build paths from it. It ensures that that the
   plateaus are disjointed, a maximum distance threshold must be
   specified.
-  ``k_shortest_paths``: This function implements the Yen algorithm for
   finding k shortest paths in a graph.

MEASURES
--------

The ``routing_measures`` module includes functions for computing various
measures related to routing:

-  ``distinct_edges_traveled``: Computes the set of distinct edges
   traveled in a list of paths.
-  ``redundancy``: Computes the redundancy score of a set of paths.
-  ``normalized_jaccard_coefficient``: Computes the normalized Jaccard
   coefficient between two sets of edges.
-  ``compute_edge_load``: Computes the load on each edge in a list of
   paths.
-  ``compute_edge_load_normalized``: Computes the load normalized by
   path cardinality on each edge in a list of paths.
-  ``get_load_balance_entropy``: Computes the entropy of the set of
   paths on the road network.
-  ``compute_edge_capacity``: Computes the capacity of each edge in a
   road network.
-  ``compute_voc``: Computes the volume-over-capacity ratio for each
   edge in a road network.
-  ``dis``: Computes the dissimilarity between two paths.
-  ``div``: Computes the diversity between a list of paths.
-  ``compute_driver_sources``: Provide a list of paths with an edge to
   tile mapping and get the traffic source tiles for each edge
-  ``compute_MDS``: Get the Major Driver Sources by filtering according
   to a threshold
-  ``compute_k_road``: Get the kroad values for each edge based on the
   MDS.
-  ``split_interval_sliding``: Get specified window intervals
-  ``compute_temp_redundancy_sliding``: Compute the Temp Redundancy of a
   certain traffic assignment result according to a specified time
   interval

UTILS
-----

The ``routing_utils`` module offers utility functions for road routing:

-  ``from_sumo_to_igraph_network``: Converts a SUMO road network to an
   igraph network.
-  ``get_shortest_path``: Finds the shortest path between two edges in a
   igraph graph and translates it to SUMO format.
-  ``compute_path_cost``: Computes the cost of a path in a graph.
-  ``test_sumo_ig_shortest_paths``: Tests the conversion of sumo to
   iGraph with the shortest path algorithm with randomly selected edges.
-  ``visualize_paths``: Visualizes paths on a map using folium.
-  ``edge_list_to_gps_list``: Converts a list of edges to a list of GPS
   points.
-  ``compute_ellipse``: Computes an ellipse representing the region
   around an edge pair.
-  ``ellipse_subgraph``: Extracts a subgraph within the region defined
   by an ellipse.
