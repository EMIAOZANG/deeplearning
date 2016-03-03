--[[
   serves train and validation data (optionally pseudo label data or any other labelled data)

   pseudolabel.t7b:
   {
      data = torch.Tensor(4000,3,96,96)
      labels = torch.Tensor(4000)
   }
]]
require 'nn'
require 'image'
require 'xlua'


torch.setdefaulttensortype('torch.FloatTensor')

-- parse STL-10 data from table into Tensor
function parseDataLabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
    t[idx]:copy(this_d[j])
    l[idx] = i
    idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end

-- class definition of Provider
local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 4000
  local valsize = 1000  -- Use the validation here as the valing set
  local channel = 3
  local height = 96
  local width = 96

  local datadir = '../dat/stl-10'

  -- download dataset
  if not paths.dirp(datadir) then
     os.execute('mkdir '..datadir)
     local www = {
         train = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/train.t7b',
         val = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/val.t7b',
         extra = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/extra.t7b',
         test = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/test.t7b'
     }

     os.execute('wget ' .. www.train .. '; '.. 'mv train.t7b '..datadir..'/train.t7b')
     os.execute('wget ' .. www.val .. '; '.. 'mv val.t7b '..datadir..'/val.t7b')
     os.execute('wget ' .. www.test .. '; '.. 'mv test.t7b '..datadir..'/test.t7b')
     os.execute('wget ' .. www.extra .. '; '.. 'mv extra.t7b '..datadir..'/extra.t7b')
  end

  local raw_train = torch.load(datadir..'/train.t7b')
  local raw_val = torch.load(datadir..'/val.t7b')
  --note that parsed_extra is a 4000*3*96*96 tensor
  local raw_pseudoLabelTrain = torch.load(datadir..'/pseudolabel.t7b')

  -- load and parse dataset
  self.trainData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return trsize end
  }

  self.trainData.data, self.trainData.labels = parseDataLabel(raw_train.data,
                                                   trsize, channel, height, width)
  
  -- concatenating train and pseudolabel set
  pdsize = #(raw_pseudoLabelTrain.data)
  self.trainData.data = torch.cat(self.trainData.data, raw_pseudoLabelTrain.data)
  self.trainData.labels = torch.cat(self.trainData.labels, raw_pseudoLabelTrain.labels)
  self.trainData.size = function() return trsize + pdsize end

  local trainData = self.trainData

  self.valData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return valsize end
  }
  self.valData.data, self.valData.labels = parseDataLabel(raw_val.data,
                                                 valsize, channel, height, width)
  local valData = self.valData

  -- convert from ByteTensor to Float
  self.trainData.data = self.trainData.data:float()
  self.trainData.labels = self.trainData.labels:float()
  self.valData.data = self.valData.data:float()
  self.valData.labels = self.valData.labels:float()
  collectgarbage()
end
