"""
PSEUDOCODE / STEPS:
1. Define 'FaissManager' Class:
   - Manages the Vector Database (FAISS) for storing content-addressable memory (Exemplars).
   - 1.1. Initialization (__init__):
          - Define paths for the index file and the metadata map.
          - Check if an index exists:
            - If YES: Load it from disk.
            - If NO: Create a new Flat L2 Index.
   - 1.2. add_vectors:
          - Adds new feature vectors to the FAISS index.
          - Stores corresponding metadata (path, class, box) in a separate JSON-compatible map.
   - 1.3. save:
          - Persists the FAISS index to disk (.index file).
          - Persists the metadata mapping to disk (.json file).
   - 1.4. get_all_exemplars:
          - Retrieves the full list of stored metadata items (used for Replay).
"""

import faiss
import numpy as np
import json
import os

# 1. Define FaissManager Class
class FaissManager:
    # 1.1. Initialization
    def __init__(self, index_path="data/memory.index", map_path="data/memory_map.json", dim=256):
        self.index_path = index_path
        self.map_path = map_path
        self.dim = dim
        
        # Load existing or create new
        if os.path.exists(self.index_path):
            print(f"Loading Faiss index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            with open(self.map_path, 'r') as f:
                self.mapping = {int(k): v for k, v in json.load(f).items()}
        else:
            print("Creating new Faiss index")
            self.index = faiss.IndexFlatL2(self.dim)
            self.mapping = {} # ID -> {'path': str, 'cls': int, 'box': [x1, y1, x2, y2]}

    # 1.2. Add Vectors
    def add_vectors(self, vectors, metadata_list):
        """
        vectors: (N, dim) numpy array
        metadata_list: list of dicts with keys 'path', 'cls', 'box'
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Vectors and metadata must have same length")
            
        start_id = self.index.ntotal
        self.index.add(vectors.astype('float32'))
        
        for i, meta in enumerate(metadata_list):
            self.mapping[start_id + i] = meta
            
    # 1.3. Save Index
    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, 'w') as f:
            json.dump(self.mapping, f)
        print(f"Saved index with {self.index.ntotal} vectors.")

    # 1.4. Retrieve All
    def get_all_exemplars(self):
        return list(self.mapping.values())
