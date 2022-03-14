class BasePipeline:
    def __init__(self, database, loss, ranking):
        self.database = database
        self.model = self.database.model
        self.loss = loss
        self.ranking = ranking
    
    def query(self, input, k=5):
        embedding = self.model(input)
        results = self.database.search(embedding, k)
        return results