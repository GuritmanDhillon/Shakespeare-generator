This code is designed to generate more of the input text. This demo uses a dataset that contains all the works of shakespeare.


Changing the input file will help change the data thats output.

The current weights are set to be extremely small so that the model can be ran on lower end devices.
To run a stronger model, you can modify the hyperparameters, specifically the
(batch_size, block_size, n_embd, n_head and n_layer)
The other parameters can be changed, but if done incorrectly, it can create problems with the model
