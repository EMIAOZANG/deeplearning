require 'nn'
require 'image'

function randSelect(dataset, nSize)
   local indices = torch.randperm(dataset:size()[1]):long()[{{1,nSize}}] -- creates tensor of random indices
   local sample = dataset:index(1,indices)
   return sample
end
