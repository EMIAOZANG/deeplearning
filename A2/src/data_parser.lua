--class definition for DataParser
--[[
   Defines a robust multi-channel image data parser class
   Methods:
      __init(numSamples, numChannels, height, width) : intializes the DataParser object, 
      parseData(d, numSamples, numChannels, height, width) : parses raw img data package(in a folded style)
      normalize():preprocess the 
]]
do
	local DataParser = torch.class('DataParser')
	
	function DataParser:__init(numSamples, numChannels, height, width)
	   self.nSamples = numSamples
	   self.nChannels = numChannels
	   self.imgHeight = height
	   self.imgWidth = width
	   self.testData = {
	      data = torch.Tensor(numSamples, numChannels, height, width),
	      -- we don't really need labels, labels = torch.Tensor(numSamples),
	      size = function() return numSamples end
	   }

	end
	

   function DataParser:parseData(d, numSamples, numChannels, height, width)
      --parse raw folded data and save to self.testData 
      local t = torch.ByteTensor(numSamples, numChannels, height, width)
      local idx = 1
	   for i = 1, #d do
	   local this_d = d[i]
	    for j = 1, #this_d do
	      t[idx]:copy(this_d[j])
	      idx = idx + 1
	    end
	end
	assert(idx == numSamples+1)
	self.testData.data = t:float()
   end

   function DataParser:parseTensorData(d)
      --[[
      Just copy the loaded tensor to self.testData.data for future processing
      ]]
      self.testData.data:copy(d) 
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

