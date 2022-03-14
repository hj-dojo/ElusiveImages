import torch
import faiss
import os
import numpy as np
from PIL import Image

class BaseDatabase:
    def __init__(self, model, folder, db=None, saveto=None):
        self.model = model
        self.folder = folder
        if db is None:
            self.db = faiss.IndexFlatL2(128)
            self.__build_db__(saveto)
        else:
            self.db = faiss.deserialize_index(np.load('db'))
        
    def __build_db__(self, saveto):
        with torch.no_grad():
            for filename in os.listdir(self.folder):
                f = os.path.join(self.folder, filename)
                img = Image.open(f)
                embedding = self.model(img)
                self.db.add(embedding)
        if saveto is not None:
            self.save(saveto)

    def search(self, input, k):
        _, I = self.db.search(input, k)
        return I

    def save(self, filename):
        serialized = faiss.serialize_index(self.db)
        np.save(filename+".npy", serialized)