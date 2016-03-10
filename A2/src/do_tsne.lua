require 'nn'
require 'image'
require 'xlua'
require 'cunn'
dofile 'tsne_visualization.lua' --import methods needed for tsne visualization
dofile 'data_parser.lua' --import data parser

opt = lapp[[
   -i, --inputFile (default '../dat/rand_1000_parsed_test.t7b')
   -o, --outputFile (default '../dat/tsne.png')
   -m, --modelPath (default '../logs_pseudolabel/model.net')
   -b, --batchSize (default 1000)
   -l, --layer (default 1)
]]
--main function
imgData = torch.load(opt.inputFile)
parser = DataParser(imgData:size(1), imgData:size(2), imgData:size(3), imgData:size(4))
parser:parseTensorData(imgData) -- save imgData to self.testData.data
parser:normalize() --normalize data for prediction

model = torch.load(opt.modelPath) -- load model
model:evaluate() --switch to evaluate mode

local imgs = parser.testData.data:cuda()

local bs = opt.batchSize
layerOuts = torch.DoubleTensor()
for t = 1, parser.testData.data:size(1), bs do
   model:forward(imgs:narrow(1,t,bs)) --batch prediction
   torch.cat(layerOuts, model.modules[opt.layer].output:double(), 1)
end

collectgarbage()

--after prediction is completed, get the output of some layer of the model
--local layerOut = model.modules[opt.layer].output:double()
print(layerOuts:size())
local tsneImage = tSNEVis(imgs, layerOuts, 4096, 2)
image.save(opt.outputFile,tsneImage)
collectgarbage()
