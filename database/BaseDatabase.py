import torch
import faiss
import os
import numpy as np
from PIL import Image
import glob


class BaseDatabase:
    def __init__(self, model, folder, transforms, imgdims=(224, 224), db=None, saveto=None, size=1000):
        self.model = model
        self.folder = folder
        self.im_indices = []
        self.transforms = transforms
        self.w, self.h = imgdims
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if db is None:
            self.db = faiss.IndexFlatL2(size)
            self.__build_db__(saveto)
        else:
            self.db = faiss.deserialize_index(np.load(db))
            self.load_im_indices()
        
    def __build_db__(self, saveto):
        print('building database')
        self.model.eval()
        with torch.no_grad():
            for f in glob.glob(os.path.join(self.folder, '*/*')):
                img = Image.open(f)
                img = img.resize((self.w, self.h))
                img = torch.unsqueeze(self.transforms(img), 0).to(self.device)
                embedding = self.model(img)
                embedding = embedding.cpu().numpy()
                self.db.add(embedding)
                self.im_indices.append(f)
        if saveto is not None:
            self.save(saveto)

    def search(self, input, k):
        with torch.no_grad():
          input = input.resize((self.w, self.h))
          #input = torch.tensor([self.transforms(input).numpy()]).to(self.device)
          input = torch.unsqueeze(self.transforms(input), 0).to(self.device)
          input = self.model(input).cpu().numpy()
          _, I = self.db.search(input, k)
        return I

    def save(self, filename):
        print("Saving database to {}".format(filename))
        serialized = faiss.serialize_index(self.db)
        np.save(filename+".npy", serialized)

    def load_im_indices(self):
          for f in glob.glob(os.path.join(self.folder, '*/*')):
              self.im_indices.append(f)



  
