----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- models:
--   + linear
--   + 2-layer neural network (MLP)
--   + convolutional network (ConvNet)
--
-- It's a good idea to run this script with the interactive mode:
-- $ th -i 2_model.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to play with the model.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('MNIST Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-poolSize', 2, 'pool size')
   cmd:option('-filtSize', 5, 'filter size')
   cmd:option('-dropoutProb',0.5,'dropout probability')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 10-class problem
noutputs = 10

-- input dimensions
nfeats = 1
width = 32
height = 32
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,128}
print('filtSize='..opt.filtSize)
print('poolSize='..opt.poolSize)
filtsize = opt.filtSize --5
poolsize = opt.poolSize --2
dropoutProb = opt.dropoutProb --0.5
owidth = width
normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.ReLU())
   model:add(nn.Linear(nhiddens,noutputs))

elseif opt.model == 'convnet' then

   -- a typical convolutional network, with locally-normalized hidden
   -- units, and L2-pooling

   -- Note: the architecture of this convnet is loosely based on Pierre Sermanet's
   -- work on the SVHN dataset (http://arxiv.org/abs/1204.3968). In particular
   -- the use of LP-pooling (with P=2) has a very positive impact on
   -- generalization. Normalization is not done exactly as proposed in
   -- the paper, and low-level (first layer) features are not fed to
   -- the classifier.

   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
   model:add(nn.ReLU())
   model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
   model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))
   owidth = torch.floor((owidth - filtsize + 1)/poolsize) -- assumes pool stride = pool width, conv kernel = 1

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialDropout(dropoutProb))
   model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
   model:add(nn.ReLU())
   model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
   model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

   owidth = torch.floor((owidth - filtsize + 1)/poolsize) -- assumes pool stride = pool width, conv kernel = 1

   -- stage 3 : standard 2-layer neural network
   model:add(nn.Reshape(nstates[2]*owidth*owidth))
   model:add(nn.Linear(nstates[2]*owidth*owidth, nstates[3]))
   model:add(nn.ReLU())
   model:add(nn.Dropout(dropoutProb))
   model:add(nn.Linear(nstates[3], noutputs))

else

   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
	 print '==> visualizing ConvNet filters'
	 print('Layer 1 filters:')
	 itorch.image(model:get(1).weight)
	 print('Layer 2 filters:')
	 itorch.image(model:get(5).weight)
      else
	 print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
