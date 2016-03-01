#TODO


##Surrogate Classes
1. run and debug data augmentation script
1. train surrogate classifier on labeled data
1. train surrogate classifier on labeled + unlabeled data
1. run labeled data through surrogate classifier, then feed last hidden layer through simple classifier
1. validate, test


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
1. write visualization script
1. use t-sne to visualize feature clusters. 
