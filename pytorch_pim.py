import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

pimadata=pd.read_csv(r"..\MVA\pima.csv", header=0).to_numpy()
X=pimadata[:,:-1]
y=pimadata[:,-1]
print(X.shape)
print(y.shape)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

print(X.shape)
print(y.shape)

model = nn.Sequential(
nn.Linear(8, 12),
nn.ReLU(),
nn.Linear(12, 8),
nn.ReLU(),
nn.Linear(8, 1),
nn.Sigmoid()
)

loss_fn = nn.BCELoss() # binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_list=[]
n_epochs = 500
batch_size = 10
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_list.append(loss)
    print(f'Finished epoch {epoch}, latest loss {loss}')


loss_list_np = [tensor.item() for tensor in loss_list]
plt.plot(np.arange(n_epochs), loss_list_np)
