import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

class ConvGNNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(ConvGNNModel, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dglnn.GraphConv(hidden_feats, hidden_feats)
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, graph, features):
        x = self.conv1(graph, features)
        x = F.relu(x)
        x = self.conv2(graph, x)
        x = self.fc(x)
        return x


d_name = 'ogbn-arxiv'
dataset = DglNodePropPredDataset(name=d_name)
graph, labels = dataset[0]
graph = dgl.add_self_loop(graph)
num_classes = dataset.num_classes
features = graph.ndata['feat']
labels = labels.squeeze()  
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

in_feats = features.shape[1]
hidden_feats = 64
out_feats = num_classes
print(f"Number of input features: {in_feats}, Number of classes: {out_feats}")
print("Number of hidden features:", hidden_feats)

conv_model = ConvGNNModel(in_feats, hidden_feats, out_feats)

loss_fn = nn.CrossEntropyLoss()
conv_optimizer = torch.optim.Adam(conv_model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    conv_model.train()
    conv_optimizer.zero_grad()
    conv_logits = conv_model(graph, features)
    conv_loss = loss_fn(conv_logits[train_idx], labels[train_idx])
    conv_loss.backward()
    conv_optimizer.step()

    

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Conv Loss: {conv_loss.item():.4f}")


def evaluate(model, graph, features, labels, idx):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        predictions = logits[idx].argmax(dim=1)
        accuracy = (predictions == labels[idx]).float().mean().item()
    return accuracy

conv_val_accuracy = evaluate(conv_model, graph, features, labels, valid_idx)
conv_test_accuracy = evaluate(conv_model, graph, features, labels, test_idx)


print(f"Conv Model Validation Accuracy: {conv_val_accuracy:.4f}, Test Accuracy: {conv_test_accuracy:.4f}")
