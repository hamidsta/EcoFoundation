import json
import matplotlib.pyplot as plt
import torch
import os 
import numpy as np
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:  # Changed logic for improvement
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

 
def plot_graph(data, title="Graph", node_color='lightblue'):
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx
    # Convert PyG data to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Plot the graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=500, font_size=10, font_weight="bold", edge_color="gray")
    plt.title(title)
    plt.show()
                
                
      

def plot_curves(base_model_save_dir: str, Save_bool = True ): 
    # Create a subdirectory for the plots
    if Save_bool :
        plot_dir = os.path.join(base_model_save_dir, 'Individual_Plots')
        os.makedirs(plot_dir, exist_ok=True)

    # Get all training history JSON files in the directory
    history_files = [f for f in os.listdir(base_model_save_dir) if (f.startswith('training_history') or f.startswith('history')) and f.endswith('.json')]

    for file in history_files:
        # Load the history
        with open(os.path.join(base_model_save_dir, file), 'r') as f:
            history = json.load(f)  
            # Create a new figure for each file
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training and validation loss on the same graph
        ax.plot(history['train_loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Set overall title with hyperparameters
        plt.suptitle(file, fontsize=12)
        
        plt.tight_layout()
        
        # Save the figure
        plot_filename = file+'.png'
        plt.savefig(os.path.join(plot_dir, plot_filename), dpi=300, bbox_inches='tight')
        
        # Display the plot in the environment output
        plt.show()
        
        # Close the figure to free up memory
        plt.close(fig)

    print(f"Individual training and validation loss curves saved in {plot_dir}")
        

def load_trained_model(model , model_path: str):
    import torch
    """
     function to load trained model
    """
    try:
        #Load checkpoint
        checkpoint = torch.load(model_path)
        

        #Load the weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        print("Model loaded ")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    
    
def train_and_evaluate_classifier(classifier, train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels):
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

    
    train_labels = train_labels.astype(int)
    val_labels = val_labels.astype(int)
    test_labels = test_labels.astype(int)
    
    # Train the classifier
    classifier.fit(train_embeddings, train_labels)
    
    # Evaluate on validation set
    val_predictions = classifier.predict(val_embeddings)
    #val_accuracy = accuracy_score(val_labels, val_predictions)
    val_f1 = f1_score(val_labels, val_predictions, average='binary') 

    
    # Evaluate on test set
    test_predictions = classifier.predict(test_embeddings)
    test_cm = confusion_matrix(test_labels, test_predictions)
    #test_accuracy = accuracy_score(test_labels, test_predictions)
    test_precision = precision_score(test_labels, test_predictions, average='binary')
    test_recall = recall_score(test_labels, test_predictions, average='binary')
    test_f1 = f1_score(test_labels, test_predictions, average='binary')
    
    # Compute AUC-ROC if the classifier has predict_proba method
    if hasattr(classifier, "predict_proba"):
        test_auc_roc = roc_auc_score(test_labels, classifier.predict_proba(test_embeddings)[:, 1])
        val_auc_score = roc_auc_score(val_labels, classifier.predict_proba(val_embeddings)[:, 1])

    else:
        test_auc_roc = None 
        val_auc_score = None 
    
    return val_auc_score, val_f1, test_cm, test_precision, test_recall, test_f1, test_auc_roc


def check_tma_objects_gene_order(tma_objects):
    """
    Checks if all AnnData objects in tma_objects have the same var_names (genes)
    in the same order.

    Parameters
    ----------
    tma_objects : list of AnnData
        List of single-cell AnnData objects to check.

    Returns
    -------
    bool
        True if all tma_objects have identical var_names in the same order,
        False otherwise.
    """
    if len(tma_objects) < 2:
        print("Only one or zero TMA object(s). Nothing to compare, returning True.")
        return True

    # Use the first TMA object's var_names as the reference
    reference_var_names = tma_objects[0].var_names

    for i, adata in enumerate(tma_objects[1:], start=1):
        current_var_names = adata.var_names
        # First check length match
        if len(reference_var_names) != len(current_var_names):
            print(f"Mismatch: TMA object #0 has {len(reference_var_names)} genes, "
                  f"but object #{i} has {len(current_var_names)} genes.")
            return False
        
        # Check exact ordering and names
        if not np.array_equal(reference_var_names, current_var_names):
            print(f"Mismatch in gene ordering/names between TMA object #0 and object #{i}!")
            return False

    # If we reach here, all var_names matched
    print("All TMA objects have the same genes in the same order.")
    return True
