import torch
import sys  # Added this import
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from torch_geometric.transforms import GDC


def mask_nodes(data, mask_ratio):
    """_summary_

    Args:
        data (_type_): Graph
        mask_ratio (_type_): ratio to mask

    Returns:
        _type_: masked graph
    """
    num_nodes = data.num_nodes
    num_masked = int(num_nodes * mask_ratio)  #  number of nodes to mask

    # Randomly select exactly num_masked nodes
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask_indices = torch.randperm(num_nodes)[:num_masked]
    mask[mask_indices] = True

    # mask each nodes features set to true
    masked_data = data.clone()
    masked_data.x[mask] = 0

   # if not torch.equal(data.x, masked_data.x) :

        # Remove edges connected to the masked nodes
        #mask_edge = mask[data.edge_index[0]] | mask[data.edge_index[1]]
        #masked_data.edge_index = data.edge_index[:, ~mask_edge]

    return masked_data
   # else :
    #    print('Masking did not work as expected, augmented sample are similar to the original')
     #   sys.exit() 
        

def simple_subgraph_sampling(data, sample_ratio):
    """
    Create a simple subgraph by sampling a subset of nodes and their connections.

    Args:
        data (torch_geometric.data.Data): Input graph
        sample_ratio (float): Ratio of nodes to keep in the subgraph

    Returns:
        torch_geometric.data.Data: Sampled subgraph
    """
    num_nodes = data.num_nodes
    num_sampled = int((num_nodes * sample_ratio) )

    # Randomly select nodes to keep
    keep_nodes = torch.randperm(num_nodes)[:num_sampled]

   # update edges
    edge_index, _, edge_mask = subgraph(keep_nodes, data.edge_index, relabel_nodes=True, num_nodes=num_nodes, return_edge_mask=True)

    # Create new data object
    subgraph_data = data.clone()
    subgraph_data.x = data.x[keep_nodes]
    subgraph_data.edge_index = edge_index
    subgraph_data.num_nodes = num_sampled  # Update the number of nodes

    return subgraph_data




def create_negative_sample(data):
    """_summary_

    Args:
        data (_type_): a graph : input shape : (x, y)

    Returns:
        _type_: fully random graph ( random edges connection ). Keep the graph with the same number of edges than data (x, y )
    """
    negative_data = data.clone()
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)

    # Generate random edges
    possible_edges = torch.combinations(torch.arange(num_nodes), r=2).t()  # All possible edges
    num_possible_edges = possible_edges.size(1)

    # Shuffle possible edges to randomize
    shuffled_edges = possible_edges[:, torch.randperm(num_possible_edges)]

    # Ensure the number of edges in the negative sample matches the original graph
    num_edges_to_sample = min(num_edges, num_possible_edges)
    new_edges = shuffled_edges[:, :num_edges_to_sample]

    # Create a new graph with the updated edges
    negative_data.edge_index = new_edges

    # Shuffle columns/ row of nodes features matrix
    if hasattr(negative_data, 'x') and negative_data.x is not None :

        row_perm = torch.randperm(negative_data.x.shape[0]) # list of random value of range =  size of data row
        columns_perm = torch.randperm(negative_data.x.shape[1]) # # list of random value of range =  size of data column
        # permut position
        negative_data.x = negative_data.x[row_perm]
        negative_data.x = negative_data.x[:,columns_perm]
    else :
        return (" error can't process further ")

    return negative_data




def node_attribute_augmentation(graph: Data, noise_scale: float = 0.01) -> Data:
    """
    Add Gaussian noise to node features.

    Args:
        graph (Data): PyTorch Geometric Data object representing the graph.
        noise_scale (float): Standard deviation of the Gaussian noise.

    Returns:
        Data: Augmented graph with perturbed node features.
    """
    augmented_graph = graph.clone()

    # Add noise to node features
    noise = torch.randn_like(augmented_graph.x) * noise_scale
    augmented_graph.x = augmented_graph.x + noise

    return augmented_graph



def calculate_avg_degree(graph_list):
    total_degree = 0
    total_nodes = 0
    for graph in graph_list:
        # Count the number of unique edges
        unique_edges = torch.unique(graph.edge_index.sort(dim=0).values, dim=1)
        num_unique_edges = unique_edges.size(1)

        # If the graph is undirected and each edge is only stored once, multiply by 2
        if num_unique_edges * 2 == graph.edge_index.size(1):
            total_degree += num_unique_edges * 2
        else:
            total_degree += graph.edge_index.size(1)

        total_nodes += graph.num_nodes

    return total_degree / total_nodes


def graph_diffusion(graph_list : list[str] , graph: Data,  alpha: float = 0.05, self_loop_weight: float = 0.2,
                    normalization_in: str = 'sym', normalization_out: str = 'col') -> Data:
        

    """
    Apply Graph Diffusion Convolution (GDC) to the graph.

    Args:
        graph_list (list) : List of graph for calculating avg degree 
        graph (Data): PyTorch Geometric Data object representing the graph.
        alpha (float): Teleportation probability in PersonalizedPageRank.
        self_loop_weight (float): Weight of the added self-loops.
        normalization_in (str): Normalization method for the input transition matrix.
        normalization_out (str): Normalization method for the output transition matrix.

    Returns:
        Data: Augmented graph after applying GDC.
    """
    avg_degree = int(calculate_avg_degree(graph_list))
    print('avg degree ', avg_degree)
    gdc = GDC(self_loop_weight=self_loop_weight,
              normalization_in=normalization_in,
              normalization_out=normalization_out,
              diffusion_kwargs={'method': 'ppr', 'alpha': alpha},
              sparsification_kwargs={'method': 'threshold', 'avg_degree': avg_degree},
              exact=True)

    return gdc(graph)