import torch, random, math, json
from torch import nn
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
        from torch import nn
        ################# Your Code Starts Here #################
        self.conv1 = torch.nn.Sequential(
           #8*15*15
            nn.Conv2d(
                in_channels=8,    # height of the input graph
                out_channels=16,  # height of the output graph
                kernel_size=3,    # 5x5 conv kernels，servel as filters
                stride=1,         # shift each kernel on the graph and scan with a stride of one
                padding=1,     # padding the graph periherals with zeors, so that the width and length is the same.
            ),
            # pass through the conv1 layer with an outcome of size (16,8,8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # after pooling, transmit (16,4,4) into next conv2 layer
        )
        ## second conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,    # the same as mentioned
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # after conv, transmit (32,4,4) into the pooling layer.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # after pooling, transmit (32,2,2) into the output layer
        )
        ## output layer    
        self.output = nn.Linear(in_features=32*2*2, out_features=1)


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
        x = self.unflatten(x)
        x = self.conv1(x)
        x = self.conv2(x)        # [batch, 32,2,2]
        x = torch.flatten(x, 1)   # conserve the batch number, and flatten the (batch, 32*7*7) into (batch,32*7*7)
        y = self.output(x)
        return y
        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

def trainmodel():
    # Well, you might want to create a model a little better than this...
    # Create an instance of NeuralNet.
    model = NeuralNet()

#     # ... and if you do, this initialization might not be relevant any more ...
#     model[1].weight.data = initialize_weights()
#     model[1].bias.data = torch.zeros(1)

    #create loss function and optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.05,weight_decay=0.001)
    
    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    for epoch in range(2000):
        for x,y in trainloader:
            y_pred=model(x)
            loss=loss_fn(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # ... after which, you should save it as "model.pkl":
    torch.save(model, 'model.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()
    
