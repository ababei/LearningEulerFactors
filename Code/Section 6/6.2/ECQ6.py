'''
This file is the main neural net training code. 
The input data used comes from a file called filtered_curves.txt, which contains all curves extracted from 
ap1e4. This file can be obtained by running ECQ6extract.py. The data consists of ap values for the first 100 primes, 
appended by the conductor and root number. The best model is saved, as well as saliency plots and the training and test curves.
One might uncomment lines of code for different use cases.
'''

import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sympy.functions
import sympy.functions.combinatorial
import sympy.functions.combinatorial.numbers
import sympy.ntheory
import sympy.ntheory.generate
import torch
import torch.nn as nn
import torch.optim as optim

p = 2
pidx = 0
for j in range(1, 101):
    if sympy.prime(j) == p:
        i = j - 1

# Define columns
cols = [str(p + 1) for p in range(100)]
cols.append('conductor')
cols.append('root number')

import torch
import sympy

print("Getting primes..")
primes = [sympy.prime(n) for n in range(1,101)]

primes = torch.tensor(primes, dtype=torch.float32)

rows = []

# Open file and process each line
file = open('filtered_curves.txt', 'r')
for line in file:
    row = [x for x in line.split(',')]
    row[0] = row[0].split('[')[1]
    row[-1] = row[-1].split(']')[0]
    row = [int(x) for x in row]
    rows.append(row)

rows_tensor = torch.tensor(rows, dtype=torch.float32)
rows_tensor = rows_tensor[-2000000:]

# Note that a2 is rows_tensor[:, 0] and a97 is rows_tensor[:, 24]
print("Normalizing each ap to be ap / sqrt(p) for all primes except the chosen one")
for i in range(0,100):
    if i == pidx:
        continue
    else:
        rows_tensor[:,i] = rows_tensor[:,i] / torch.sqrt(primes[i])

#for i in range(25,100):
#    rows_tensor[:,i] = rows_tensor[:,i] / torch.sqrt(primes[i])

# The training input is all rows except the chosen prime row
X = torch.cat((rows_tensor[:, :pidx], rows_tensor[:, pidx+1:25], rows_tensor[:, 100:]), dim=1)
y = rows_tensor[:, pidx]

y = y.int()
y = y.tolist()
y_one_hot = [a2 % 2 for a2 in y]

# dividing the conductor by 10^6 (in the case where I train on the first 25 primes)
X[:, 24] = X[:, 24]/1e+6 

# Making the number of 0s and 1s equal in the distribution
print("Making the distribution equal..")
check = [int(x) for x in y_one_hot]
ind = torch.nonzero(torch.tensor(y_one_hot))[-500000:].squeeze(1)
full = torch.arange(1730087)

# Remove the numbers in ind
tensor_filtered = full[~torch.isin(full, ind)][-500000:]
tensor_filtered.shape
full_tensor = torch.cat((tensor_filtered, ind))
z = torch.tensor([y_one_hot[i] for i in full_tensor])
Xz = [X[i] for i in full_tensor]

Xz = torch.stack(Xz)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from collections import Counter

print("Scaling training dataset..")
# normalize training data to be mean 0 variance 1:
scaler_X = StandardScaler()
training_data_normalized = scaler_X.fit_transform(Xz)

from sklearn.model_selection import train_test_split
import os

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training_data_normalized, z, test_size=10000, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create TensorDataset and DataLoader for the training data
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

train_dataset = TensorDataset(X_train, y_train)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Defining the neural net
class MultiBinaryClassifier(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim):
        super(MultiBinaryClassifier, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip([input_dim, *layer_dims], [*layer_dims, output_dim])
        ])
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # We don't want to ReLU the last layer
            if i != len(self.layers) - 1:
                x = nn.functional.relu(x)
                x = self.dropout(x)
        return torch.sigmoid(x).squeeze(1)
    
input_dim = X_train.shape[1]
# Writing the parameters of the neural net
hidden_dims = [128, 64, 32, 2]
output_dim = 1 
learning_rate = 0.0001
weight_decay = 0.02
epochs = 30000 

model = MultiBinaryClassifier(input_dim, hidden_dims, output_dim).to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for multi-label classification
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Set up TensorBoard
writer = SummaryWriter()

train_loss = np.zeros(epochs)
test_loss = np.zeros(epochs)
best_acc = 0.0

print("Starting training..")
for epoch in range(epochs):
    for raw_mats, expected in train_loader:
        raw_mats.to(device)
        expected.to(device)

        optimizer.zero_grad()
        output = model(raw_mats)
        loss = criterion(output, expected.to(torch.float32))
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        train_loss[epoch] = float(criterion(model(X_train), y_train))
        test_loss[epoch] = float(criterion(model(X_test), y_test))
    
    if epoch % 10 == 1: 
        print(f"Epoch {epoch:4}: Train loss {train_loss[epoch]:.4f}, Test loss {test_loss[epoch]:.4f}")
        writer.add_scalars("loss", {'train':train_loss[epoch],'test':test_loss[epoch]}, epoch)
        writer.flush()
        if epoch == 1:
            best_acc = test_loss[epoch]
        else:
            # Saving the best model
            if test_loss[epoch] < best_acc:
                out_path = os.path.join("models/", f"modeleulerfaca{p}mod2.pt")
                print(f"test loss {test_loss[epoch]} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_acc = test_loss[epoch]

print("Training done.")



from sklearn.metrics import matthews_corrcoef

# Assuming `model` is your trained model and `X_test`, `y_test` are your data
model.eval()  # Set the model to evaluation mode

# Forward pass (without logsoftmax)
logits = model(X_test)

# Detach the logits from the computation graph and convert to numpy
logits = logits.detach().numpy()

# Assuming logits are the raw outputs (before any activation functions like sigmoid)
predicted = (logits > 0.5).astype(int)  # For binary classification (you can adjust for multiclass)

# Calculate MCC
mcc = matthews_corrcoef(y_test, predicted)
print(f"Matthews Correlation Coefficient: {mcc}")

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

print("Generating test and train loss picture")
plt.plot(range(epochs), test_loss, 'o', color='orange')
plt.plot(range(epochs), train_loss, 'o', color='blue')
plt.xlabel("Epochs")
plt.title(f"Train vs test loss for a{p} mod 2")

# Show the plot
plt.show()
plt.savefig(f"models/TrainTesta{p}mod23")
print("Done")

print("Computing saliency..")
# Prepare the input tensor
input_tensor = torch.tensor(X_test, dtype=torch.float32)
input_tensor.requires_grad_()  # Enable gradient computation

# Initialize a tensor to accumulate saliencies
accumulated_saliency = torch.zeros_like(input_tensor[:,:99])
accumulated_saliency1 = torch.zeros_like(input_tensor)

# Forward pass through the model
output = model(input_tensor)

# Iterate over each sample in X_test
for i in range(len(X_test)):
    # Compute the gradient with respect to the input for the current sample
    output[i].backward(retain_graph=True)  # backward through the chosen class

    # Get the gradients for the current sample
    gradients = input_tensor.grad.detach().numpy()

    # Calculate saliency (absolute value of gradients) for the current sample
    saliency = np.abs(gradients[i, :99])  # Only for the current sample
    saliency1 = np.abs(gradients[i])

    # Accumulate the saliency values
    accumulated_saliency[i] = torch.tensor(saliency)
    accumulated_saliency1[i] = torch.tensor(saliency1)

    # Zero the gradients for the next sample
    input_tensor.grad.zero_()

# Calculate the mean saliency across all samples
mean_saliency = accumulated_saliency.mean(dim=0).detach().numpy()  # Mean across all samples
mean_saliency1 = accumulated_saliency1.mean(dim=0).detach().numpy()  # Mean across all samples

#print(mean_saliency)
# Plot the heatmap of the mean saliency
#plt.imshow(mean_saliency.reshape(1, -1), cmap='hot', aspect='auto', interpolation='nearest')
plt.plot(mean_saliency, 'o')
#plt.colorbar()  # Add a colorbar to show the gradient magnitude
plt.xlabel("Input Features")
plt.title(f"Mean Saliency Map for a{p} mod 2")

# Show the plot
# plt.show()
plt.savefig(f"models/MeanSaliencyMapa{p}mod23")

plt.plot(mean_saliency1, 'o')
#plt.colorbar()  # Add a colorbar to show the gradient magnitude
plt.xlabel("Input Features")
plt.title(f"Mean Saliency Map with conductor for a{p} mod 2")

# Show the plot
plt.show()
plt.savefig(f"models/MeanSaliencyMapwConda{p}mod23")
