import torch, random, math, json
import torch.nn as nn
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE=torch.float32
DEVICE=torch.device("cpu")

###########################################################################################
class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################
        self.conv1 = nn.Sequential(
           #15*8*8
            nn.Conv2d(
                in_channels=15,    # height of the input graph
                out_channels=20,  # height of the output graph
                kernel_size=3,    # 5x5 conv kernels，servel as filters
                stride=1,         # shift each kernel on the graph and scan with a stride of one
                padding=1,     # padding the graph periherals with zeors, so that the width and length is the same.
            ),
            # pass through the conv1 layer with an outcome of size (30,8,8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # after pooling, transmit (30,4,4) into next conv2 layer
        )
        ## second conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,    # the same as mentioned
                out_channels=40,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # after conv, transmit (60,4,4) into the pooling layer.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # after pooling, transmit (60,2,2) into the output layer
        )
        ## output layer    
        self.output = nn.Linear(in_features=40*2*2, out_features=1)


        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        if (len(x.shape)==1):
            x=x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x)        # [batch, 32,2,2]
        x = torch.flatten(x, 1)   # conserve the batch number, and flatten the (batch, 32*7*7) into (batch,32*7*7)
        y = self.output(x)
        return y.squeeze()
        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

def trainmodel():
#     model = torch.nn.Sequential(
#             nn.Conv2d(
#                 in_channels=15,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=64,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Flatten(),
#             nn.Linear(in_features=64 * 2 * 2, out_features=256),
#             nn.ReLU(),
#             nn.Linear(in_features=256, out_features=1),
#         )

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=8*8*15, out_features=512),
        torch.nn.BatchNorm1d(num_features=512), 
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_features=512, out_features=128),
        torch.nn.BatchNorm1d(num_features=128), 
        torch.nn.Sigmoid(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 1)
         )



    # Create loss function and optimizer.
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    
    # Load the training data.
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    
    # Train the model.
    n=2000
    for epoch in range(n):
#         print("Epoch #", epoch)
        for x, y in trainloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)  # modify the loss function to compare single value           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    # Save the trained model.
    torch.save(model, 'model.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()
    
