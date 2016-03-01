require 'torch'
require 'image'
require 'xlua'
require 'cunn'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

--class definition for DataParser
do
	local DataParser = torch.class('DataParser')
	
	function DataParser:__init(numSamples, numChannels, height, width)
	   self.nSamples = numSamples
	   self.nChannels = numChannels
	   self.imgHeight = height
	   self.imgWidth = width
	   self.testData = {
	      data = torch.Tensor(numSamples, numChannels, height, width),
	      labels = torch.Tensor(numSamples),
	      size = function() return numSamples end
	   }

	end
	
	function DataParser:parseDataLabel(d, numSamples, numChannels, height, width)
	   --parse labels from raw data, and save the parsed data to self.testData
	   local t = torch.ByteTensor(numSamples, numChannels, height, width)
	   local l = torch.ByteTensor(numSamples)
	   idx = 1
	   print(#d)
	   for i = 1, #d do --multiple data folds, images in the same fold has same label
	      local this_d = d[i]
	      print("this_d size:"..#this_d)
	      for j = 1, #this_d do --this_d is a image record 
	         t[idx]:copy(this_d[j])
	         l[idx] = i
	         idx = idx + 1
	      end
	   end
	   assert(idx == numSamples + 1) --check if we have imported all images
	   self.testData.data = t:float()
	   self.testData.labels = l:float()
	end
	
	function DataParser:normalize()
	   local testData = self.testData
	   print 'preprocessing data (color space + normalization)'
	   collectgarbage()
	
	   --preprocess testSet
	   local normalization = nn.SpatialContrastiveNormalization(1,image.gaussian1D(7))
	   for i = 1, testData:size() do
	      xlua.progress(i, testData:size())
	      local rgb = testData.data[i]
	      local yuv = image.rgb2yuv(rgb)
	      -- normalize y locally:
	      yuv[1] = normalization(yuv[{{1}}])
	      testData.data[i] = yuv
	   end
	   --normalize
	   local mean_u = testData.data:select(2,2):mean()
	   local std_u = testData.data:select(2,2):std()
	   testData.data:select(2,2):add(-mean_u)
	   testData.data:select(2,2):div(std_u)
	   -- normalize v globally:
	   local mean_v = testData.data:select(2,3):mean()
	   local std_v = testData.data:select(2,3):std()
	   testData.data:select(2,3):add(-mean_v)
	   testData.data:select(2,3):div(std_v)
	 
	   testData.mean_u = mean_u
	   testData.std_u = std_u
	   testData.mean_v = mean_v
	   testData.std_v = std_v
	end
end
--ending class definition

--

--prediction function definition
function predict(modelPath, testPath, height, width)
   -- loads model and runs it on test dataset to generate prediction.csv
   local model = torch.load(modelPath)
   collectgarbage()
   local rawTestData = torch.load(testPath)
   local dataProvider = DataParser(8000, 3, 96, 96)
   print(#rawTestData.data)
   dataProvider:parseDataLabel(rawTestData.data, 8000, 3, 96, 96)
   dataProvider:normalize()

   model:evaluate()
   
   print('==> getting predictions from test set:')
   print((dataProvider.testData):size())

   local preds = torch.Tensor(dataProvider.testData.data:size(1)) 
print(preds:size())
   collectgarbage()
   
   local bs = 25

   --only mini-batch supported?
   for t = 1, (dataProvider.testData.data):size(1), bs do
print("t="..t)

      --local miniBatch = torch.CudaTensor(1,(dataProvider.testData.data):size(2),(dataProvider.testData.data):size(3), (dataProvider.testData.data):size(4))
      --print('miniBatch size:'..miniBatch:size())
      local input = dataProvider.testData.data:cuda()
      --miniBatch[{{1},{},{},{}}]=input
      print(input:size())
      --local pred = model:forward(miniBatch)
      local pred, idx = torch.max(model:forward(input:narrow(1,t,bs)),2)

print(idx:size())
      preds[{{t,t+bs-1}}] = idx:int()
      collectgarbage()
   end
   return preds
end

function save_pred_file(fname, preds)
   --save list of predictions to file name
   --
   --[[
      args:
         fname: file name of save file
         preds: tensor of predictions
      returns:
         non, saves to disk
   ]]

   local file = io.open(fname, "w")
   file:write("Id,Prediction\n")
   io.close(file)

   local file = io.open(fname,"a") 

   for t = 1, preds:size()[1] do
      file:write(t..','..preds[t] .. "\n")
   end
   io.close(file)
end

mPath = "log/sample/model.net"
tPath = "stl-10/test.t7b"
predictions = predict(mPath, tPath, 96, 96)
save_pred_file("predictions.csv", predictions)
