import os
import sys
import glob
import scanpy
import pickle

# Add the directory containing the function folder to the Python path
sys.path.append("C:/Users/hsta1/OneDrive/Bureau/AI_notebook/EcoFoundation_Hamid_Last/EcoFoundation_Hamid")

# Custom unpickler to handle the module path mismatch
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "utils":
            from function.utils import EcoFoundationObject
            return EcoFoundationObject
        return super().find_class(module, name)

def get_tma_objects(base_path="C:/Users/hsta1/OneDrive/Bureau/AI_notebook/EcoFoundation_Hamid_Last/EcoFoundation_Hamid/data/processed/"):
    """
    Load TMA h5ad objects from a local folder.
    Returns a list of Scanpy AnnData objects.
    """
    pattern = os.path.join(base_path, "Processed_new_TMA_*.h5ad")
    tma_files = sorted(glob.glob(pattern))
    tma_objects = []
    
    for file in tma_files:
        try:
            tma = scanpy.read_h5ad(file)
            tma_objects.append(tma)
            print(f"Loaded: {os.path.basename(file)}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    print(f"\nTotal TMA objects loaded: {len(tma_objects)}")
    return tma_objects

def get_eco_objects(base_path="C:/Users/hsta1/OneDrive/Bureau/AI_notebook/EcoFoundation_Hamid_Last/EcoFoundation_Hamid/data/processed/"):
    """
    Load EcoFoundation objects (pickled files).
    Returns a list of EcoFoundationObject (graph) instances.
    """
    pattern = os.path.join(base_path, "Eco_TMA_*.plk")
    eco_files = sorted(glob.glob(pattern))
    eco_objects = []
    
    for file in eco_files:
        try:
            with open(file, 'rb') as f:
                eco = CustomUnpickler(f).load()
            eco_objects.append(eco)
            print(f"Loaded: {os.path.basename(file)}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    print(f"\nTotal ECO objects loaded: {len(eco_objects)}")
    return eco_objects