import igraph
from igraph import Graph
import sumolib
import numpy as np
import folium
import warnings
import seaborn as sns

from tqdm.notebook import tqdm

from typing import List



""" utilities for road routing """


def from_sumo_to_igraph_network(road_network):
    
    """
    Converts a SUMO road network to an igraph network.

    Parameters:
    -----------
    road_network : SUMO road network
        A SUMO road network object.

    Returns:
    --------
    G : igraph graph
        An igraph graph representing the road network.
    """
    
    
    edges = []
    nodes = set()

    edges_attr = {"id":[], "length":[], "speed_limit":[], "traveltime":[]}

    
    # Iterate over all the (regular) edges in the road network
    for edge in road_network.getEdges():


        # The ID of the from and to nodes of the edge will be:
        # <edge_id>_from -> <edge_id>_to
        id_from = edge.getID()+"_from"
        id_to = edge.getID()+"_to"

        # add edge & attributes
        edges.append([id_from, id_to])

        edges_attr["id"].append(edge.getID())
        edges_attr["length"].append(edge.getLength())
        edges_attr["speed_limit"].append(edge.getSpeed())
        edges_attr["traveltime"].append(edge.getLength()/edge.getSpeed())

        # add nodes to the set
        nodes.add(id_from)
        nodes.add(id_to)

        
    # create the connections

    # Iterate over all nodes in the road network
    for node in road_network.getNodes():

        # Get the connections of the node
        connections = list(node.getConnections())

        # Iterate over all connections of the node
        for conn in connections:

            # Get the IDs of the from and to edges
            from_edge_conn = conn.getFrom().getID()+"_to"
            to_edge_conn = conn.getTo().getID()+"_from"


            # add edge & attributes
            edges.append([from_edge_conn, to_edge_conn])

            edges_attr["id"].append("connection")
            edges_attr["length"].append(0)
            edges_attr["speed_limit"].append(-1)
            edges_attr["traveltime"].append(0)

            # add nodes to the set
            nodes.add(from_edge_conn)
            nodes.add(to_edge_conn)
            
            
     # create the Igraph Graph
    
    G = Graph(directed=True)
    G.add_vertices(list(nodes))
    G.add_edges(edges, edges_attr)
    
    return G



def get_shortest_path(G, from_edge, to_edge, attribute):
    """
    Find the shortest path between two edges in a igraph graph, and translate it to SUMO format.

    Parameters:
        G: The igraph graph
        from_edge: ID of the edge where the path starts
        to_edge: ID of the edge where the path ends
        optimize: The edge attribute to optimize the path on.

    Returns:
        A dictionary with the following keys:
        - 'sumo' (List[str]): The edges of the path in SUMO format
        - 'ig' (List[str]): The igraph edges ID of the path
        - 'cost' (float): The total cost of the path
    """
    index_from = G.vs.find(name=f"{from_edge}_from").index
    index_to = G.vs.find(name=f"{to_edge}_to").index
    
    path = G.get_shortest_paths(index_from, index_to, weights=attribute, output="epath")
    
    edges_ig = path[0]
    
    total_cost = compute_path_cost(G, edges_ig, attribute)

    edges_sumo = [e for e in G.es[path[0]]["id"] if e != "connection"]

    return {"sumo": edges_sumo, "ig": edges_ig, "cost": total_cost}




def compute_path_cost(G: igraph.Graph, edge_list: List[str], attribute: str) -> float:     
    """
    This function is used to compute the cost of a path in a graph.

    Parameters:
    G (igraph graph): An igraph graph representing the road network.
    edge_list (list): A list of edges that form the path.
    attribute (str): The key to use to compute the cost, the key must be in the edge_data

    Returns:
    total_cost (float): The total cost of the path
    """
        
    total_cost = sum([e for e in G.es[edge_list][attribute] if e != "connection"])

    return total_cost




def test_sumo_ig_shortest_paths(road_network_sumo, G, n_tests, attribute, th=1e-4):
    
    
    sumo_edges = road_network_sumo.getEdges()
    
    fastest = True if attribute == "traveltime" else False
    
    for i in tqdm(range(n_tests)):
    
        from_edge = sumo_edges[np.random.randint(0,len(sumo_edges))]
        to_edge = sumo_edges[np.random.randint(0,len(sumo_edges))]
        
        res_sumo = road_network_sumo.getOptimalPath(from_edge, to_edge, fastest=fastest)
        
        if res_sumo[0] is not None:
            edges_sumo = []
            for e in res_sumo[0]:
                edges_sumo.append(e.getID())

            cost_sumo = res_sumo[1]

            res_ig = get_shortest_path(G, from_edge, to_edge, attribute)

            edges_sumo_ig = res_ig["sumo"]
            cost_ig = res_ig["cost"]


            if len(edges_sumo_ig)==0 and len(edges_sumo)==0:
                continue

            assert(abs(cost_ig-cost_sumo)<th)
            
            
            


def visualize_paths(path_list, road_network, colors=None, opacity=1, map_f=None, dotted = False, show_markers = False):
        
    if colors is None:
        color_list = ["blue"]*len(path_list)
    elif isinstance(colors, str) and colors!="rand":
        color_list = [colors]*len(path_list)
    elif colors == "rand":
        color_list = sns.color_palette("colorblind", len(path_list)).as_hex()
        
        
    # transform to GPS points
    paths_gps = [edge_list_to_gps_list(edge_list, road_network) for edge_list in path_list]

    if map_f is None:
        map_f = folium.Map(location=[paths_gps[0][1][1], paths_gps[0][1][0]], tiles='cartodbpositron', zoom_start=13)

    for path, col in zip(paths_gps, color_list): 
          
          folium.PolyLine(locations=[list(reversed(coord)) for coord in path], 
                          weigth=4, color=col, opacity=opacity, 
                          dash_array='10' if dotted else None).add_to(map_f)

    if show_markers:
        lon_from, lat_from = paths_gps[0][0]
        lon_to, lat_to = paths_gps[0][-1]
        folium.Marker(location=[lat_from, lon_from], icon = folium.Icon(color='lightgreen'), popup='ORIGIN').add_to(map_f)
        folium.Marker(location=[lat_to, lon_to], icon = folium.Icon(color='lightred'), popup='TARGET').add_to(map_f)

    return map_f


def edge_list_to_gps_list(edge_list, road_network):
    
    gps_points = []

    for edge_id in edge_list:
        
        sumo_edge = road_network.getEdge(edge_id)

        x, y = sumo_edge.getFromNode().getCoord()
        lon_from, lat_from = road_network.convertXY2LonLat(x, y)

        x, y = sumo_edge.getToNode().getCoord()
        lon_to, lat_to = road_network.convertXY2LonLat(x, y)

        gps_points.append((lon_from, lat_from))
        gps_points.append((lon_to, lat_to))
        
        
    return gps_points
