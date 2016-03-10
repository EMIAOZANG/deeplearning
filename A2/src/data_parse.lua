--[[
This file is used to parse the STL-10 dataset into one Tensor.
By Jake Zhao @NYU
For DS-GA-1008 2016 Spring
--]]


numSamples = 8000
numChannels = 3
height = 96
width = 96

function parseData(d)
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
   return t
end


-- Usage example, parsing the test data
test_data = torch.load('../dat/stl-10/test.t7b')
parse_test_data = parseData(test_data.data)
torch.save('parsed_test.t7b', parse_test_data)
