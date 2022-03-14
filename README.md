# ElusiveImages

## Pipeline
An image retrieval pipeline consists of: A database, an image embedding module (same as used for your database), and a ranking module.
The pipeline supports the `query(input, k)` method, which returns the top k images based on your input.

## Database
A database consists of: Your images, and an image embedding module. It stores the embeddings for each element of our dataset using the faiss module for quick lookup. If you would like to save the embeddings to avoid recomputing them later, you can save it using the `saveto` variable which will save it to an .npy file, and later on load it using the `db` variable. You can call the `query(input, k)` method, which returns the top k images based on your input.

Critically, the `query` method should utilize whatever ranking function you choose -- the example in BaseDatabase uses L2 distance from the encoded images via `faiss.search()`.
## Models
These are the models used for image embeddings. They can be trained using `train.py` in the parent folder, with configs in the `config` folder, datasets in the `dataset` folder,
and loss function in the `loss` folder.
## Loss
Fairly self-explanatory -- define your loss functions for models here.
## Dataset
Here you can define the datasets for training the models. They are meant to inherit from pytorch's `Dataset` class to make it easy to train with them.