#Imports
import torch
import torch.nn as nn  #all nn modules
import torch.nn.functional as F  #all activation funcs such as relu,tanh
import torch.optim as optim #all optim such as gradient desc ,etc
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation


#Create Fully connected Network
# class NN(nn.Module): #nn module is used to train and build the layers of neural networks such as input, hidden, and output.
#     def __init__(self,input_size,num_classes): #size if 28*28 = 784
#         super(NN, self).__init__() #calls the constructor of the parent so that any initialization done in the superclass is still done
#         self.fc1 = nn.Linear(input_size,50)
#         self.fc2 = nn.Linear(50,num_classes) #it takes 50 nodes to num_classes which are 0-9
#
#     def forward(self,x):
#         """
#                 x here is the mnist images and we run it through fc1, fc2 that we created above.
#                 we also add a ReLU activation function in between and for that (since it has no parameters)
#                 I recommend using nn.functional (F)
#                 Parameters:
#                     x: mnist images
#                 Returns:
#                     out: the output of the network
#         """
#         x = F.sigmoid(self.fc1(x))
#         x = self.fc2(x)
#         return x
#         # model = NN(784,10)
#         # x = torch.randn(64,784)
#         # print(model(x).shape)


# Create simple CNN
class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10): # in channel is 1 as it is bnw otherwise 3(RGB)
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        # Nout = [(Nin + 2p - k)/s] - 1
        #thats why it is called Same convulation
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) # makes it half(14*14)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7,num_classes) # 7*7 as we r using 2 maxpool layers

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# to save the model
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# to load the model
def load_checkpoint(checkpoint,model,optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
load_model = True

# Load Datasets
train_dataset = datasets.MNIST(root='dataset/',train=True,transform = transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset = datasets.MNIST(root='dataset/',train=False,transform = transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#Initialize a Network
model = CNN(in_channels=in_channels,num_classes = num_classes).to(device)

#Loss Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
checkpoint_path = "my_checkpoint.pth.tar"
if load_model:
    load_checkpoint(torch.load(checkpoint_path),model,optimizer)

#Train Network
for epoch in range(num_epochs):
    losses = []

    if(epoch == 2):
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # print(data.shape)

        # Get to correct shape
        #data = data.reshape(data.shape[0],-1) #to make it to a vector (already done in forward)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

    mean_losses = sum(losses) / len(losses)
    print(f'Loss at epoch {epoch}: {mean_losses}')

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking Accuracy of Training set')
    else:
        print('Checking Accuracy of Test set')
    """
        Check accuracy of our trained model given a loader and a model
        Parameters:
            loader: torch.utils.data.DataLoader
                A loader for the dataset you want to check accuracy on
            model: nn.Module
                The model you want to check accuracy on
        Returns:
            acc: float
                The accuracy of the model on the dataset given by the loader
    """
    num_correct = 0
    num_samples = 0
    model.eval()
    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:
            #Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            #Get to correct shape
            #x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)
            # Check how many we got correct
            num_correct += (predictions == y).sum()
            # Keep track of number of samples
            num_samples += predictions.size(0)  #64

        model.train()
        return num_correct/num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")