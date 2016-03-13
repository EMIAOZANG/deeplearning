--Run images through the given model, and output requested hidden layer output
require 'torch'
require 'cunn'
require 'image'
require 'nn'
require 'xlua'
require 'optim'
local c = require 'trepl.colorize' --prints in color!

model_path = "logs_pseudolabel/model.net"
batch_size = 10
layer_num = 1
out_path = "../dat/output_pl.t7"

dofile('../src/provider.lua')
provider = torch.load('../dat/provider.t7') --load provider data
provider.trainData.data = provider.trainData.data:float() --convert to float
image_batch = provider.trainData.data[{{1,batch_size}}]
print("image batch size:", image_batch:size())
function get_output(image_batch, model, layer_num)
  model:forward(image_batch:cuda())
  local pool_nodes = model:findModules('nn.SpatialMaxPooling')
  print("pool nodes:")
  print(pool_nodes)
  local output = pool_nodes[layer_num].output:double()
  return output
end

model = torch.load(model_path)
print("full model:")
print(model)
output = get_output(image_batch, model, layer_num)
print(output:size())
torch.save(out_path,output)
