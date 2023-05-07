from routing_utils import *
from routing_measures import dis, div 
import itertools

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
            result_list.append({"edges": sp_k["sumo"], "original_cost": original_cost, "penalized_cost": sp_k["cost"]})
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



