import torch
import matplotlib.pyplot as plt
# Y = 0.5*X^3 - 0.8*X^2 + 0.3*X^0
X = torch.rand((1000,1),dtype=torch.float32)
t1 = X**3
t2 = X**2
t3 = X**0
X = torch.cat((t1,t2,t3),dim=1)
coefficient_actual = torch.tensor((0.5,-0.8,0.3))
Y = X @ coefficient_actual.view(-1,1)
W = torch.rand((3,1), dtype=torch.float32, requires_grad=True)

batch_size = 32
lr = 0.1
n = X.shape[0]
epochs = []
losses = []
for epoch in range(1000):
    perm = torch.randperm(n)

    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]

        Xb = X[idx]
        Yb = Y[idx]

        Y_pred = Xb @ W
        loss = ((Y_pred - Yb)**2).mean() / 2
        losses.append(loss.item())
        epochs.append(epoch)
        print("Weight: ", W)
        print("Loss: ", loss)
        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad
        W.grad.zero_()

plt.figure()
plt.plot(epochs, losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.show()