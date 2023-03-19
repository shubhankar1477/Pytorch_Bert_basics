import torch

class IMDBDataset:
    def __init__(self,reviews,targets):
        self.reviews = reviews
        self.targets = targets

    def __len__(self):
        return len( self.reviews)
    
    def __getitem__(self,idx):
        review = self.reviews[idx,:]
        target = self.target[idx]

        return {
            "review":torch.tensor(review,dtype=torch.long),
             "target":torch.tensor(target,dtype=torch.long)

        }

        