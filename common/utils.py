import pickle
import numpy as np 
import json
import os
import sys
import yaml
import copy
import logging
import ast
import random
from datetime import datetime

from common.registry import Registry
import world


# def process_knowledge(path):

#     with open(path, "r") as file:
#         # Read the contents
#         content = file.read()
#     elements = content.split()

#     new_form = "["+', '.join(elements).replace(".,", ".0,").replace("[,", "[")+"]"
#     new_data = ast.literal_eval(new_form) # here the data is not in correct order: [cars: 38.    phase: 0.    acc: 1.    dec: 2.   E_dec: 2.5   delay: 0.4   weather: 0.99]
#     # later in the dataset, it will be correct into: [cars: 38.    phase: 0.   weather: 0.99    dec: 2.   E_dec: 2.5   delay: 0.4    acc: 1.]
#     new_data = np.array(new_data)
#     strench = new_data.reshape(-1, new_data.shape[-1])
#     strench[:,[2, 6]] = strench[:,[6, 2]] # finish change

#     return strench


def get_weather(x=0.99, test=False):

    choice_name = ['heavy snowy', 'snowy', 'rainy', 'sunny'] 
    list_data = [0.25, 0.5, 0.75, 0.99]
    result = x
    if test:
        inde = list_data.index(x)
        weather_type = choice_name[inde]
        return weather_type, result
    # Important!!!
    # remember to change this if the settings is changed!!!
    else:
        # Validate and ensure x is either 0.75 or 0.5
        # ---
        # if x not in [0.99, 0.75, 0.5]:
        #     raise ValueError("x should be either 0.75 or 0.5")

        # inde = list_data.index(x)

        # values = [0.25, 0.5, 0.75, 0.99]
        # probabilities = [0.1, 0.1, 0.1, 0.1]  # 70% probability for x, 30% for other values
        # probabilities[inde] = 0.7

        # # Discretize the samples based on the probabilities
        # result = random.choices(values, weights=probabilities)[0]

        # weather_type = choice_name[inde]
        # ---
        result = x
        inde = list_data.index(x)
        weather_type = choice_name[inde]


        # If True, sample x, otherwise sample one of the other three values
        return weather_type, result

# def get_weather(insight):
#     choice = ['heavy snowy', 'snowy', 'rainy', 'sunny'] 
#     value = [0.25, 0.5, 0.75, 0.99]
#     index = random.randint(0, len(choice)-1)
#     weather_type = choice[index]
#     weather_value = value[index]

#     return weather_type, weather_value

def get_road_dict(roadnet_dict, road_id):
    for item in roadnet_dict['roads']:
        if item['id'] == road_id:
            return item
    raise KeyError("environment and graph setting mapping error, no such road exists")


def build_index_intersection_map(roadnet_file):
    """
    generate the map between identity ---> index ,index --->identity
    generate the map between int ---> roads,  roads ----> int
    generate the required adjacent matrix
    generate the degree vector of node (we only care the valid roads(not connect with virtual intersection), and intersections)
    return: map_dict, and adjacent matrix
    res = [net_node_dict_id2inter,net_node_dict_inter2id,net_edge_dict_id2edge,net_edge_dict_edge2id,
        node_degree_node,node_degree_edge,node_adjacent_node_matrix,node_adjacent_edge_matrix,
        edge_adjacent_node_matrix]
    """
    roadnet_dict = json.load(open(roadnet_file, "r"))
    virt = "virtual" # judge whether is virtual node, especially in convert_sumo file
    if "gt_virtual" in roadnet_dict["intersections"][0]:
        virt = "gt_virtual"
    valid_intersection_id = [node["id"] for node in roadnet_dict["intersections"] if not node[virt]]
    node_id2idx = {}
    node_idx2id = {}
    edge_id2idx = {}
    edge_idx2id = {}
    node_degrees = []  # the num of adjacent nodes of node

    edge_list = []  # adjacent node of each node
    node_list = []  # adjacent edge of each node
    sparse_adj = []  # adjacent node of each edge
    invalid_roads = []
    cur_num = 0
    # build the map between identity and index of node
    for node_dict in roadnet_dict["intersections"]:
        if node_dict[virt]:
            for node in node_dict["roads"]:
                invalid_roads.append(node)
            continue
        cur_id = node_dict["id"]
        node_idx2id[cur_num] = cur_id
        node_id2idx[cur_id] = cur_num
        cur_num += 1
    # map between identity and index built done

    # sanity check of node number equals intersection numbers
    if cur_num != len(valid_intersection_id):
        raise ValueError("environment and graph setting mapping error, node 1 to 1 mapping error")
    
    # build the map between identity and index and built the adjacent matrix of edge
    cur_num = 0
    for edge_dict in roadnet_dict["roads"]:
        edge_id = edge_dict["id"]
        if edge_id in invalid_roads:
            continue
        else:
            edge_idx2id[cur_num] = edge_id
            edge_id2idx[edge_id] = cur_num
            cur_num += 1
            input_node_id = edge_dict['startIntersection']
            output_node_id = edge_dict['endIntersection']
            input_node_idx = node_id2idx[input_node_id]
            output_node_idx = node_id2idx[output_node_id]
            sparse_adj.append([input_node_idx, output_node_idx])
    
    # build adjacent matrix for node (i.e the adjacent node of the node, and the 
    # adjacent edge of the node)
    for node_dict in roadnet_dict["intersections"]:
        if node_dict[virt]:
            continue        
        node_id = node_dict["id"]
        road_links = node_dict['roads']
        input_nodes = []  # should be node_degree
        input_edges = []  # needed, should be node_degree
        for road_link_id in road_links:
            road_link_dict = get_road_dict(roadnet_dict, road_link_id)
            if road_link_dict['endIntersection'] == node_id:
                if road_link_id in edge_id2idx.keys():
                    input_edge_idx = edge_id2idx[road_link_id]
                    input_edges.append(input_edge_idx)
                else:
                    continue
                start_node = road_link_dict['startIntersection']
                if start_node in node_id2idx.keys():
                    start_node_idx = node_id2idx[start_node]
                    input_nodes.append(start_node_idx)
        if len(input_nodes) != len(input_edges):
            raise ValueError(f"{node_id} : number of input node and edge not equals")
        node_degrees.append(len(input_nodes))

        edge_list.append(input_edges)
        node_list.append(input_nodes)

    node_degrees = np.array(node_degrees)  # the num of adjacent nodes of node
    sparse_adj = np.array(sparse_adj)  # the valid num of adjacent edges of node

    result = {'node_idx2id': node_idx2id, 'node_id2idx': node_id2idx,
              'edge_idx2id': edge_idx2id, 'edge_id2idx': edge_id2idx,
              'node_degrees': node_degrees, 'sparse_adj': sparse_adj,
              'node_list': node_list, 'edge_list': edge_list}
    return result



def analyse_vehicle_nums(file_path):
    replay_buffer = pickle.load(open(file_path, "rb"))
    observation = [i[0] for i in replay_buffer]
    observation = np.array(observation)
    observation = observation.reshape([-1])
    print("the mean of vehicle nums is ", observation.mean())
    print("the max of vehicle nums is ", observation.max())
    print("the min of vehicle nums is ", observation.min())
    print("the std of vehicle nums is ", observation.std())


def get_output_file_path(task, model, prefix):
    path = os.path.join('./data/output_data', task, model, prefix)
    return path


def load_config(path, previous_includes=[]):
    if path in previous_includes:
        raise ValueError(
            f"Cyclic configs include detected. {path} included in sequence {previous_includes}."
        )
    previous_includes = previous_includes + [path]

    direct_config = yaml.load(open(path, "r"), Loader=yaml.Loader)

    # Load configs from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, '{}' provided".format(type(includes))
        )

    config = {}
    duplicates_warning = []
    duplicates_error = []

    # TODO: Need test duplication here
    for include in includes:
        include_config, inc_dup_warning, inc_dup_error = load_config(
            include, previous_includes
        )
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config, merge_dup_error = merge_dicts(config, include_config)
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning += merge_dup_warning

    return config, duplicates_warning, duplicates_error


def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py
    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.
    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


def build_config(args):
    # configs file of specific agents is loaded from configs/agents/{agent_name}
    agent_name = os.path.join('./configs/agents', args.task, f'{args.agent}.yml')
    config, duplicates_warning, duplicates_error = load_config(agent_name)
    if len(duplicates_warning) > 0:
        logging.warning(
            f"Overwritten configs parameters from included configs "
            f"(non-included parameters take precedence): {duplicates_warning}"
        )
    if len(duplicates_error) > 0:
        raise ValueError(
            f"Conflicting (duplicate) parameters in simultaneously "
            f"included configs: {duplicates_error}"
        )
    args_dict = vars(args)
    for key in args_dict:
        config.update({key: args_dict[key]})  # short access for important param

    # add network(for FRAP and MPLight)
    cityflow_setting = json.load(open(config['path'], 'r'))
    config['traffic']['network'] = cityflow_setting['network'] if 'network' in cityflow_setting.keys() else None
    return config
