from routing_utils import *
from routing_measures import dis, div 
import itertools

# KSPML and KSPMO
from queue import PriorityQueue
from collections import Counter

import warnings 



def apply_penalization(G, from_edge, to_edge, k, attribute, fun_2_apply, arguments, apply_to="sp_edges", all_distinct=True, remove_tmp_attribute=True, max_iter=1e3):


    # number of iterations
    it=0

    # path list is used to check for duplicates while result_list stores the k paths and their original and penalized cost
    result_list, path_list = [], []
    
    # Create a copy of the attribute (e.g., traveltime -> tmp_traveltime)
    G.es[f"tmp_{attribute}"] = G.es[attribute]  
       
    while len(result_list)<k and it<max_iter:

        # compute the shortest path on the copy of the attribute
        sp_k = get_shortest_path(G, from_edge, to_edge, f"tmp_{attribute}")

        # add to the result_list and path_list
        if not(all_distinct and sp_k["sumo"] in path_list):

            original_cost = compute_path_cost(G, sp_k["ig"], attribute)
            result_list.append({"edges": sp_k["sumo"], "ig" : sp_k["ig"], "original_cost": original_cost, "penalized_cost": sp_k["cost"]})
            path_list.append(sp_k["sumo"])


        # update edge weights according to the specified function        
        if apply_to == "sp_edges":
            edge_list = G.es[sp_k["ig"]]
        elif apply_to == "all_edges":
            edge_list = G.es

        fun_2_apply(edge_list, f"tmp_{attribute}", **arguments)

        it+=1
        
        
    # check for warnings
    if it == max_iter:
        warnings.warn(f'Iterations limit reached, returned {len(result_list)} distinct paths (instead of {k}).', RuntimeWarning)
    
        
    # remove the copy of the attribute from all the edges
    if remove_tmp_attribute:
        del(G.es[f"tmp_{attribute}"])
        
        
    return result_list


def path_penalization(G, from_edge, to_edge, k, p, attribute, all_distinct=True, remove_tmp_attribute=True, max_iter=1e3):
    
    # define the function to use to penalize the edge weights
    def update_edge_weights_pp(edge_list, attribute, p=0):
        for e in edge_list:
            if e["id"] != "connection":
                e[attribute]*=(1+p)
                
    # arguments beyond edge_list and attribute (that are mandatory)
    dict_args = {"p": p}
            
    apply_to = "sp_edges"
    
    result_list = apply_penalization(G, from_edge, to_edge, k, attribute, update_edge_weights_pp, dict_args, 
                                     apply_to=apply_to, all_distinct=all_distinct, 
                                     remove_tmp_attribute=remove_tmp_attribute, max_iter=max_iter)
    
    return result_list





def graph_randomization(G, from_edge, to_edge, k, delta, tau, attribute, all_distinct=True, remove_tmp_attribute=True, max_iter=1e3):
    
    # a more efficient implementation
    # 1. draw a list of values in N(0,1) (faster for list)
    # 2. convert each r in N(0,1) as it has been drawn from N(0,s) as following:
    # 3. new_value = r*s
    # 4. it is equivalent to draw new_value from N(0,s)
    
    def update_edge_weights_gr(edge_list, attribute, delta=0, tau=0, default_attribute=""):
        
        rand_noise_list = np.random.normal(0, 1, size=len(edge_list))
        
        for ind, e in enumerate(edge_list):
            if e["id"] != "connection":
                rand_noise = rand_noise_list[ind]*((e[default_attribute]**2)*(delta**2))
                e[attribute] = max(e[default_attribute] + rand_noise, tau)
  
                
    # arguments beyond edge_list and attribute (that are mandatort)
    dict_args = {"delta": delta, "tau": tau, "default_attribute": attribute}
    
    
    apply_to = "all_edges"
    
    result_list = apply_penalization(G, from_edge, to_edge, k, attribute, update_edge_weights_gr, dict_args, 
                                     apply_to=apply_to, all_distinct=all_distinct, 
                                     remove_tmp_attribute=remove_tmp_attribute, max_iter=max_iter)
    
    return result_list



def path_randomization(G, from_edge, to_edge, k, delta, tau, attribute, all_distinct=True, remove_tmp_attribute=True, max_iter=1e3):
    
    # define the function to use to penalize the edge weights
  
    def update_edge_weights_gr(edge_list, attribute, delta=0, tau=0, default_attribute=""):
        
        rand_noise_list = np.random.normal(0, 1, size=len(edge_list))
        
        for ind, e in enumerate(edge_list):
            if e["id"] != "connection":
                rand_noise = rand_noise_list[ind]*((e[default_attribute]**2)*(delta**2))
                e[attribute] = max(e[default_attribute] + rand_noise, tau)
  
                
    # arguments beyond edge_list and attribute (that are mandatort)
    dict_args = {"delta": delta, "tau": tau, "default_attribute": attribute}
    
    
    apply_to = "sp_edges"
    
    result_list = apply_penalization(G, from_edge, to_edge, k, attribute, update_edge_weights_gr, dict_args, 
                                     apply_to=apply_to, all_distinct=all_distinct, 
                                     remove_tmp_attribute=remove_tmp_attribute, max_iter=max_iter)
    
    return result_list


def duarouter(G, from_edge, to_edge, k, w, attribute, all_distinct=True, remove_tmp_attribute=True, max_iter=1e3):
    
    # define the function to use to penalize the edge weights
    def update_edge_weights_w(edge_list, attribute, w=1, default_attribute=""):

        rand_noise_list = np.random.uniform(1, w, size=len(edge_list))
        
        for ind, e in enumerate(edge_list):
            if e["id"] != "connection":
                rand_noise = rand_noise_list[ind]
                e[attribute] = e[default_attribute] * rand_noise
  
                
    # arguments beyond edge_list and attribute (that are mandatory)
    dict_args = {"w": w, "default_attribute": attribute}

    
    apply_to = "all_edges"
    
    result_list = apply_penalization(G, from_edge, to_edge, k, attribute, update_edge_weights_w, dict_args, 
                                     apply_to=apply_to, all_distinct=all_distinct, 
                                     remove_tmp_attribute=remove_tmp_attribute, max_iter=max_iter)
    
    return result_list



def k_disjointed(G, from_edge, to_edge, k, attribute, all_distinct=True, remove_tmp_attribute=True, max_iter=1e3):
    
    # define the function to use to penalize the edge weights
    def update_edge_weights_pp(edge_list, attribute, p=0):
        for e in edge_list:
            if e["id"] != "connection":
                e[attribute]*=(1+p)
                
    # arguments beyond edge_list and attribute (that are mandatory)
    dict_args = {"p": 1e6}
    
    apply_to = "sp_edges"
    
    result_list = apply_penalization(G, from_edge, to_edge, k, attribute, update_edge_weights_pp, dict_args, 
                                     apply_to=apply_to, all_distinct=all_distinct, 
                                     remove_tmp_attribute=remove_tmp_attribute, max_iter=max_iter)
    
    return result_list



def no_randomization(G, from_edge, to_edge, attribute):
    
    # define the function to use to penalize the edge weights
    def update_edge_weights_nr(edge_list, attribute, p=0):
        for e in edge_list:
            if e["id"] != "connection":
                e[attribute]*=(1+p)
                
    # arguments beyond edge_list and attribute (that are mandatory)
    dict_args = {"p": 0}
    
    apply_to = "sp_edges"
    
    result_list = apply_penalization(G, from_edge, to_edge, 1, attribute, update_edge_weights_nr, dict_args, 
                                     apply_to=apply_to, all_distinct=True, 
                                     remove_tmp_attribute=True, max_iter=1e3)
    
    return result_list




def k_mdnsp(G, from_edge, to_edge, k, epsilon, attribute, remove_tmp_attribute=True):

    def findsubsets_with_element(s, n, element):
        return list([c for c in itertools.combinations(s, n) if element in c])
    
    path_nsp = []
    P_kmdnsp = []
    
    G.es[f"tmp_{attribute}"] = G.es[attribute] 

    sp_k = get_shortest_path(G, from_edge, to_edge, attribute)

    path_nsp.append(sp_k["ig"])

    l_max = (1+epsilon)*sp_k["cost"]

    m = 0
    f = 2-m*((1-epsilon)/2)
    
    div_pkmdnsp = 0

    while f>1:

        # recalculate penalities
        for p1 in path_nsp:
                for edge in p1:
                    G.es[edge][f"tmp_{attribute}"] = G.es[edge][attribute]*f

        sp = get_shortest_path(G, from_edge, to_edge, f"tmp_{attribute}")
            
        p = sp["ig"]
        p_cost = compute_path_cost(G, p, attribute)
                
        if (p_cost <= l_max) and (not(p in path_nsp)):

            path_nsp.append(p)

            candidate_subsets = findsubsets_with_element(path_nsp, k, p)

            for P in candidate_subsets:
                    
                div_p = div(G, P, attribute)
                
                if div_p > div_pkmdnsp:
                    P_kmdnsp = P
                    div_pkmdnsp = div(G, P_kmdnsp, attribute)
                    #print(div_pkmdnsp, f, 2-(m+1)*((1-epsilon)/2))
                
        else:            
            m += 1
            f = 2-m*((1-epsilon)/2)
            
            
    result_list = []
    
    for path in P_kmdnsp:

        edge_list_sumo = [e for e in G.es[path]["id"] if e != "connection"]
        original_cost = compute_path_cost(G, path, attribute)
        result_list.append({"edges": edge_list_sumo, "original_cost": original_cost, "penalized_cost": -1})
    

    # remove the copy of the attribute from all the edges
    if remove_tmp_attribute:
        del(G.es[f"tmp_{attribute}"])
        
    # check for warnings
    if 0<len(result_list)<k:
        warnings.warn(f'Returned {len(result_list)} distinct paths (instead of {k}).', RuntimeWarning)
    if len(result_list) == 0:
        warnings.warn(f'No paths found (try to increase epsilon) Returning the fastest path only.', RuntimeWarning)
        result_list.append({"edges": sp_k["sumo"], "original_cost": sp_k["cost"], "penalized_cost": sp_k["cost"]})
   
        
    return result_list


# Function to compute the next simple single-via path
def next_ssvp_by_length(G, from_edge, to_edge, attribute, Q):

    if type(from_edge) != str or type(to_edge) != str:
        from_edge, to_edge = from_edge.getID(), to_edge.getID()

    def get_path_cost(G, edge_idxs):
        return sum(G.es[edge_idxs][attribute])

    def get_vertex_sumo(G, edge, pos):
        return G['edge_vertices'][edge][pos]

    def get_shortest_path_dict(vertex, mode):
        sps_dict = dict()
        sps = G.get_shortest_paths(vertex, mode = mode, weights=attribute, output = 'epath')
        
        ordering = -1 if mode == 'in' else 1

        for p in sps:
            # this way it's always it's indexed to the varying destination edge
            if p != []:
                last_edge = p[-1]
                sps_dict[last_edge] = p[::ordering]
            
        return sps_dict

    first_call = True

    # extract the from and to verteces of the origin and destination edges
    s = get_vertex_sumo(G, from_edge, pos = 'to')
    t = get_vertex_sumo(G, to_edge, pos = 'from')

    # set of shortest paths from origin to every other node indexed by the node
    Ts_to_N = get_shortest_path_dict(s, mode = 'out')
    # set of shortest paths from every node to destination indexed by the node 
    TN_to_t = get_shortest_path_dict(t, mode = 'in')

    try:
        # retrieving the shortest path (the one going to t)
        psp = Ts_to_N[G['edge_sumo_ig'][G.vs[t]['name']]]
    except KeyError:
        # if it is just after a crossing it is too difficult to retrieve the correct edge
        psp = G.get_shortest_paths(G['edge_vertices'][from_edge]['to'], 
                                   G['edge_vertices'][to_edge]['from'], 
                                   weights=attribute, output="epath")[0]
    
    yield psp
    
    # Ts_to_N.keys() & TN_to_t.keys() is a check for reached edges
    for n_edge in Ts_to_N.keys() & TN_to_t.keys():

        # check for edge not in shortest path
        not_in_sp = n_edge != s and n_edge != t and n_edge not in psp

        if not_in_sp:
            # the shortest path from s to n and its length
            ps_to_n = Ts_to_N[n_edge]
            len_ps_to_n = get_path_cost(G, ps_to_n)

            # the shortest path from n to t and its length
            pn_to_t = TN_to_t[n_edge]
            len_pn_to_t = get_path_cost(G, pn_to_t)

            # add the node, the lengths, the current vertex and the paths
            Q.put((len_ps_to_n + len_pn_to_t, n_edge, ps_to_n[:-1] + pn_to_t))

    del Ts_to_N, TN_to_t

    while not Q.empty():

        path_length, n_edge, ps_n_pn_t = Q.get()
        
        # if the path is not simple some vertex are repeated
        simple_counter = Counter(ps_n_pn_t) # I count the frequency
        

        
        try:
            # if there is no repeated vertex yield the path
            if max(simple_counter.values()) == 1: 
                yield ps_n_pn_t
                
            else: # otherwise remove the edges in between the repetition

                # get the edge indexes repeating themeselves
                freq_edge = [x for x, count in simple_counter.items() if count > 1]

                # find the first edge repetition
                loop_edge = sorted(freq_edge, key = lambda x: ps_n_pn_t.index(x))[0]

                # set the first slicing to that vertex position in the path list
                start_index = ps_n_pn_t.index(loop_edge)
                # get the last position of that vertex
                end_index = len(ps_n_pn_t) - ps_n_pn_t[::-1].index(loop_edge) - 1
                # slice off the elements in between
                ps_n_pn_t = ps_n_pn_t[:start_index] + ps_n_pn_t[end_index+1:]

                # get the path length
                path_length = get_path_cost(G, ps_n_pn_t)

                # re-add the path to the queue
                Q.put((path_length, loop_edge, ps_n_pn_t))

        # This error comes when the path is empty, I return
        except ValueError:
            yield []

    return None


def kspml(G, from_edge, to_edge, k, theta, attribute, max_iter = 1000):

    if type(from_edge) != str or type(to_edge) != str:
        from_edge, to_edge = from_edge.getID(), to_edge.getID()

    # just the opposite of the dissimilarity
    def Sim(pi, pj):
        return 1 - dis(G, pi, pj, attribute)

    PkDPwML = list()

    # initialize a priority queue (it will be based on length)
    Q = PriorityQueue()

    ssvpbl = next_ssvp_by_length(G, from_edge, to_edge, attribute, Q)

    p = next(ssvpbl)

    if p == []: 
        # just in the case the from and to edges where not directly connected
        return [{'edges' : [], 'ig' : [], 'original_cost' : 0, 'penalized_cost' : 0}]

    iter = 0

    while p is not None and len(PkDPwML) < k and iter <= max_iter:
        iter += 1
        # if the similarity is lower than theta for any two paths
        sim_condition = list()

        for p_prime in PkDPwML:
            try:
                p_p_sim = Sim(p, p_prime) < theta
                sim_condition.append(p_p_sim)

            # It means the path are identical (it may happen with path of 3 edges or less)
            except ZeroDivisionError: 
                sim_condition.append(False)
                break
            
            if not p_p_sim:
                break

        if all(sim_condition):
            PkDPwML.append(p)

        try:
            p = next(ssvpbl)
        except StopIteration: # when the generator ends
            p = None
            
    del Q
    del ssvpbl

    # to output as in the framework
    PkDPwML_List = list()

    for path in PkDPwML:
        path = [G['edge_sumo_ig'][from_edge]]+path+[G['edge_sumo_ig'][to_edge]]
        epath = G.es[path]
        path_dict = dict()
        path_dict['edges'] = list(filter(lambda x: x != 'connection', epath['id']))
        path_dict['ig'] = path
        path_dict['original_cost'] = sum(epath[attribute])
        path_dict['penalized_cost'] = path_dict['original_cost']
        PkDPwML_List.append(path_dict)

    return PkDPwML_List


def kspmo(G, from_edge, to_edge, k, theta, attribute, max_iter = 1000):

    if type(from_edge) != str or type(to_edge) != str:
        from_edge, to_edge = from_edge.getID(), to_edge.getID()

    # just the opposite of the dissimilarity

    def sim_5(G, path0, path1, attribute):

        intersection = set(path0).intersection(path1)
        sum_intersection = sum(G.es[intersection][attribute])
        
        l1 = sum(G.es[path0][attribute])
        l2 = sum(G.es[path1][attribute])

        return sum_intersection/(min(l1, l2))

    def Sim(pi, pj):
        return sim_5(G, pi, pj, attribute)

    PkDPwML = list()
    
    # initialize a priority queue (it will be based on length)
    Q = PriorityQueue()

    ssvpbl = next_ssvp_by_length(G, from_edge, to_edge, attribute, Q)

    p = next(ssvpbl)

    if p == []: 
        # just in the case the from and to edges where not directly connected
        return [{'edges' : [], 'ig' : [], 'original_cost' : 0, 'penalized_cost' : 0}]

    iter = 0

    while p is not None and len(PkDPwML) < k and iter <= max_iter:
        iter += 1
        # if the similarity is lower than theta for any two paths
        sim_condition = list()

        for p_prime in PkDPwML:
            try:
                p_p_sim = Sim(p, p_prime) < theta
                sim_condition.append(p_p_sim)

            # It means the path are identical (it may happen with path of 3 edges or less)
            except ZeroDivisionError: 
                sim_condition.append(False)
                break
            
            if not p_p_sim:
                break

        if all(sim_condition):
            PkDPwML.append(p)

        try:
            p = next(ssvpbl)
        except StopIteration: # when the generator ends
            p = None

    del Q
    del ssvpbl

    # to output as in the framework
    PkDPwML_List = list()

    for path in PkDPwML:
        path = [G['edge_sumo_ig'][from_edge]]+path+[G['edge_sumo_ig'][to_edge]]
        epath = G.es[path]
        path_dict = dict()
        path_dict['edges'] = list(filter(lambda x: x != 'connection', epath['id']))
        path_dict['ig'] = path
        path_dict['original_cost'] = sum(epath[attribute])
        path_dict['penalized_cost'] = path_dict['original_cost']
        PkDPwML_List.append(path_dict)

    return PkDPwML_List
