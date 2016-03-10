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
   -b, --batchSize (default 25)
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

for t = 1, parser.testData.data:size(), bs do
   model:forward(imgs:narrow(1,t,bs)) --batch prediction
end

--after prediction is completed, get the output of some layer of the model
local layerOut = model.modules[opt.layer].output
print('Output from the '..opt.layer..'-th layer has shape: '..layerOut:size())
local tsneImage = tSNEVis(imgs, layerOut, 4096, 2)
image.save(opt.outputFile,tsneImage)



