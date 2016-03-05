#TODO

##Questions for Lab
1. how do I connect surrogate classifier to final classifier?  I think just import the model and hit model:forward(inputs); but then how do I get out the last hidden layer?

##Surrogate Classes
1. bring script from Convolve.ipynb to run labeled data through surrogate classifier to output feature vector
1. build svm classifier.  Torch should have a simple svm criterion:
	* linearSVM = nn.Sequential()
	* linearSVM:add(nn.Linear(ninputs, 1))
	* criterion = nn.MarginCriterion()
1. validate, test, tweak models


##Pseudo-Labels
1. get two subsets of unlabeled: 1 of 4000 samples, other of 8000 samples.

  local indices = torch.randperm(dataset:size()[1]):long():split(4000) -- creates tensor of random indices

  sample = dataset:index(1,indices[1])

  torch.save()

1. (optional) augment labeled data
1. predict on unlabeled data
1. save predictions as labels
1. train on labeled + pseudo-labeled data
1. validate, test

#Write-up
1. answer questions 1 and 2
1. write visualization script
1. use t-sne to visualize feature clusters. 
