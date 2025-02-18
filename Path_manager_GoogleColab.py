import sys
import subprocess
import os
import glob
import pickle


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "utils":
            from function.utils import EcoFoundationObject
            return EcoFoundationObject
        return super().find_class(module, name)



class PathManager:
    def __init__(self):        
        # Add paths
        sys.path.append('/content/drive/MyDrive/LabMembers/Hamid/Github_Eco_Foundation/Eco_Foundation')
        sys.path.append('/content/drive/MyDrive/LabMembers/Hamid/Github_Eco_Foundation/Eco_Foundation/function')



        # Install required packages
        self.install_packages()
        
        # Base paths
        self.base_path = '/content/drive/MyDrive/LabMembers/Hamid/Eco_Foundation/Eco_Foundation/data/Processed'

    def install_packages(self):
        packages = [
            'scanpy',
            'torch',
            'torch_geometric',
            'networkx',
            'torchinfo',
            'tqdm',
            'tangram-sc',
            'squidpy', 
            'scvi-tools'  
            ]
        
        for package in packages:
            try:
                subprocess.check_call(['pip', 'install', package])
                print(f"Successfully installed {package}")
            except Exception as e:
                print(f"Error installing {package}: {e}")

    def load_tma_objects(self):
        import scanpy
        """Load TMA h5ad objects"""
        pattern = os.path.join(self.base_path, "Processed_new_TMA_*.h5ad")
        print('taking from :', pattern)
        tma_files = sorted(glob.glob(pattern))
        print(tma_files)
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
    
    def load_eco_objects(self):
        """Load EcoFoundation objects"""

        pattern = os.path.join(self.base_path, "Eco_TMA_*.plk")
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
