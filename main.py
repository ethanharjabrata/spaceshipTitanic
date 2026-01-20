import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.nn import Sequential, Linear, ReLU, BCEWithLogitsLoss
from torch.optim import Adam
from torch import from_numpy, sigmoid
from sklearn.model_selection import train_test_split

#no particular reason for splitting this into seperate files except for the sake of neatness
from preprocessing import preprocessing_and_cleaning
from helpers import set_seed


def main():
  #Cleaning and Preprocessing
  train, test = preprocessing_and_cleaning()
  y=train['Transported']
  x=train.drop(columns=['Transported','PassengerId'])
  x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.2, random_state=100)
  x_test=test.drop(columns=['PassengerId'])
  # MLP TRAINING
  #Set seed for reproducability
  set_seed(100)
  #In the off chance my GPU has a meltdown before this code is run
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #Convert to Tensors and then move to GPU
  x_train_tensor=from_numpy(x_train.values)
  #Cast from Double (default) to float so PyTorch doesn't go insane
  x_train_tensor=x_train_tensor.to(device=device).float()
  y_train_tensor=from_numpy(y_train.values)
  #convert to 2D shape (admittedly still a 1D vector) so PyTorch doesn't go insane
  y_train_tensor=y_train_tensor.to(device=device).float().view(-1,1)

  x_val_tensor=from_numpy(x_val.values)
  x_val_tensor=x_val_tensor.to(device=device).float()
  y_val_tensor=from_numpy(y_val.values)
  y_val_tensor=y_val_tensor.to(device=device).float().view(-1,1)

  #2 layer so my comuputer doesn't have a meltdown
  # print(y_train_tensor.size())
  model = Sequential(
    Linear(x_train.shape[1], 50, device=device),
    ReLU(),
    Linear(50, 1, device=device),
  )
  #set reduction to mean so that I don't have to mess with the learning rate
  loss_func=BCEWithLogitsLoss(reduction="mean")
  epochs=200
  learning_rate=1e-2

  #early stopping
  best_loss=float("inf")
  best_model_weights=None
  #allow for 10 increasingly inacurrate weights before stopping
  max_count=10
  count=0
  optimizer= Adam(params=model.parameters(), lr=learning_rate)
  for i in range(epochs):
    #Training
    model.train()
    y_pred=model(x_train_tensor)
    loss=loss_func(y_pred, y_train_tensor)
    # print(f"epoch: {i}, loss: {loss.item():.5f}")
    #Zero gradients to prevent new calculated gradient being added to old gradient from previous epochs
    optimizer.zero_grad()
    loss.backward()
    #Update the model using partial derivates calculated during backward pass
    optimizer.step()

    #Evaluation against validation set
    model.eval()
    with torch.no_grad():
      val_pred=model(x_val_tensor)
      val_loss=loss_func(val_pred, y_val_tensor)
    # print(f"epoch: {i}, train loss: {loss.item():.5f}, val loss: {val_loss.item():.5f}")
    if val_loss<best_loss:
      best_loss=val_loss
      best_model_weights=model.state_dict()
      count=0
    else:
      count+=1
      if count>=max_count:
        print(f"__EARLY STOP, EPOCH {i}__")
        model.load_state_dict(best_model_weights)
        break

  #Get final predictions
  with torch.no_grad():
    outputs= model(x_val_tensor)
    #scale from regression model to classification model
    probs=sigmoid(outputs)
    #round up or down to find True/False
    tensor_predictions=(probs>0.5)
    predictions=tensor_predictions.cpu().detach().numpy()
  #Accuracy 80.6% with 171 epochs and lr 0.01
  print(f"Accuracy: {accuracy_score(y_val, predictions)}")

  x_test_tensor=from_numpy(x_test.values)
  x_test_tensor=x_test_tensor.to(device=device).float()
  # Create submission
  with torch.no_grad():
    outputs= model(x_test_tensor)
    #scale from regression model to classification model
    probs=sigmoid(outputs)
    #round up or down to find True/False
    tensor_predictions=(probs>0.5)
    predictions=tensor_predictions.cpu().detach().numpy()
  predictions=pd.DataFrame(predictions)
  submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions[predictions.columns[0]]})
  #True Accuracy: 79.8%
  submission.to_csv('data/MLPSubmission.csv', index=False) 
  
main()