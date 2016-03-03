--[[
   Takes a random sample from unfolded data
   Usage: th random_sample.lua --size <sampleSize> --fname <filePath>
]]

require 'xlua'
require 'optim'
require 'nn'

opt = lapp[[
  -p, --prefix (default "extra")  
  -f, --fname (default "stl-10/extra.t7b")
  -s, --size (default 4000)
]]

numChannels = 3
height = 96
width = 96

print(opt)

function randSampling(d, size) 
   --print(#(d.data[1]))
   local indices = torch.randperm(#(d.data[1]))[{{1,size}}]
   
   indices:int()
   print(indices[1])
   local t = torch.ByteTensor(size, numChannels, height, width)
   print(t:size())
   print(#(d.data[1]))
   for i = 1, size do
      --print((d.data[1][indices[i]]):size())
      --print(t[i]:size())
      t[i]:copy(d.data[1][indices[i]])
   end
   return t
end

--import unlabelled data and output sampled .t7 files
inputData = torch.load(opt.fname)
outputData = randSampling(inputData, opt.size)
torch.save('parsed_'..opt.prefix..'.t7b', outputData)

