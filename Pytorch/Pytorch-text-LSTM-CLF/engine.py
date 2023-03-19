import torch 
import torch.nn as nn

def train(data_loader,model,optimiser,device):
    ## set model to train mode
     model.train()
     for data in data_loader:
          reviews = data['review']
          target = data['target']
          reviews = reviews.to(device,dtype = torch.long)
          target = target.to(device,dtype = torch.float)

          # clear gradients
          optimiser.zero_grad()

          # MAKE PREDICTION
          predictions = model(reviews)

          ## calculate loss
          loss = nn.BCEWithLogitsLoss()(
               predictions,
               target.view(-1,1)
          )
          # compute gradient w.r.t to all parameters
          loss.backward()
          #optimisation step
          optimiser.step()
def evaluate(data_loader,model,device):
     final_predictions = []
     final_targets = []
     model.eval()
     with torch.no_grad():
          for data in data_loader:
               reviews = data['review']
               target = data['target']
               reviews = reviews.to(device,dtype = torch.long)
               target = target.to(device,dtype = torch.float)
               predictions = model(reviews)
               predictions = predictions.cpu().numpy().tolist()
               targets = data['target'].cpu().numpy().tolist()
               final_predictions.extend(predictions)
               final_targets.extend(targets)
     return final_predictions,final_targets





