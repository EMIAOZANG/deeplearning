----------------------------------------------------------------------
-- This script demonstrates how to load the (MNIST) Handwritten Digit 
-- training data, and pre-process it to facilitate learning.
--
-- It's a good idea to run this script with the interactive mode:
-- $ th -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('MNIST Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'small', 'how many samples do we load: small | full')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-sub', true, 'use test data if set to true')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> downloading dataset'

-- Here we download dataset files. 

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

data_path = '../dat/mnist.t7'
train_file = paths.concat(data_path, 'train_32x32.t7')
test_file = paths.concat(data_path, 'test_32x32.t7')

if not paths.filep(train_file) or not paths.filep(test_file) then
   os.execute('wget ' .. tar .. ' -P ../dat')
   os.execute('tar xvf ' .. paths.concat('../dat/',paths.basename(tar)) .. ' -C ../dat/')
end

----------------------------------------------------------------------
-- training/test/validation size

if opt.size == 'full' then
   print '==> using regular, full training data'
   trsize = 40000
   tesize = 10000
   valsize = 20000
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trsize = 4000
   tesize = 1000
   valsize = 2000 -- adding validation set, default train:test:val ratio is 4:1:2
end

print('sub=')
print(opt.sub)
if opt.sub == true then --non-submission version
   trsize = trsize + valsize
else
   tesize = valsize
end

----------------------------------------------------------------------
print '==> loading dataset'

-- load and split data
loaded = torch.load(train_file, 'ascii')

-- create a shuffling vector
function shuffleAndSplitTrain(trsize_, valsize_, data_)
   local shuffleIndex = torch.randperm(data_.data:size(1))
   local numTrain = trsize_
   local numVal = valsize_
   local train = torch.Tensor(numTrain, data_.data:size(2), data_.data:size(3), data_.data:size(4))
   local val = torch.Tensor(numVal, data_.data:size(2), data_.data:size(3), data_.data:size(4))
   local trainLabels = torch.Tensor(numTrain)
   local valLabels = torch.Tensor(numVal)
   for i=1,numTrain do
      train[i] = data_.data[shuffleIndex[i]]:clone()
      trainLabels[i] = data_.labels[shuffleIndex[i]]
   end
   for i=1,numVal do
      val[i] = data_.data[shuffleIndex[i]]:clone()
      valLabels[i] = data_.labels[shuffleIndex[i]]
   end
   trainShuffle = {
      data = train,
      labels = trainLabels,
      size = function() return trsize_ end
   }
   valShuffle = {
      data = val,
      labels = valLabels,
      size = function() return valsize_ end
   }
   return trainShuffle, valShuffle
end

if opt.sub == false then --training version
   trainData, testData = shuffleAndSplitTrain(trsize,valsize,loaded)
else
   trainData = {
      data = loaded.data,
      labels = loaded.labels,
      size = function() return trsize end
   }
   print('tesize='..tesize)
	loaded = torch.load(test_file, 'ascii')
	testData = {
	   data = loaded.data,
	   labels = loaded.labels,
	   size = function() return tesize end
	}
end



----------------------------------------------------------------------
print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

trainData.data = trainData.data:float()
testData.data = testData.data:float()

print(trainData)
print(trainData:size())
print(testData)
print(testData:size())
-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.

-- Convert all images to YUV

-- As we are using MNIST which only has one channel, ignore the above paragraph

-- Normalize each channel, and store mean/std.
-- These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize globally'
mean = trainData.data[{ {},1,{},{} }]:mean()
std = trainData.data[{ {},1,{},{} }]:std()
trainData.data[{ {},1,{},{} }]:add(-mean)
trainData.data[{ {},1,{},{} }]:div(std)

-- Normalize test data, using the training means/stds
testData.data[{ {},1,{},{} }]:add(-mean)
testData.data[{ {},1,{},{} }]:div(std)

-- Local normalization
print '==> preprocessing data: normalize locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(7) -- normalization kernel

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for i = 1,trainData:size() do
   trainData.data[{ i,{1},{},{} }] = normalization:forward(trainData.data[{ i,{1},{},{} }])
end

for i = 1,testData:size() do
   testData.data[{ i,{1},{},{} }] = normalization:forward(testData.data[{ i,{1},{},{} }])
end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

trainMean = trainData.data[{ {},1 }]:mean()
trainStd = trainData.data[{ {},1 }]:std()

testMean = testData.data[{ {},1 }]:mean()
testStd = testData.data[{ {},1 }]:std()

print('training data mean: ' .. trainMean)
print('training data standard deviation: ' .. trainStd)
print('training data size: ' .. trainData:size())

print('test data mean: ' .. testMean)
print('test data standard deviation: ' .. testStd)
print('testing data size: ' .. testData:size())

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if itorch then
      first256Samples = trainData.data[{ {1,256} }]
      itorch.image(first256Samples)
   else
      print("For visualization, run this script in an itorch notebook")
   end
end
