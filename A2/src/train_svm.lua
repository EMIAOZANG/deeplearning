require 'xlua'
require 'optim'
require 'cunn'
dofile './provider.lua'
local c = require 'trepl.colorize' --prints in color!

opt = lapp[[
   -s,--save                  (default "logs/svm")      subdirectory to save logs
   -b,--batchSize             (default 64)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default sample)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --num_targets              (default 10)       number of surrogate classes
   --patch_size               (default 32)        size of patch to take from image
   --surrogate_path           (default ./logs/surrogate2/model.net)  path to surrogate classifier
]]

print(opt)

--build svm model
print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda()) -- model shift to 'cuda' mode
model:add(nn.View(512*2*2)) --TODO: make this a variable
model:add(nn.Linear(512*2*2, opt.num_targets):cuda()) --TODO: make this a variable or extract from surrogate_model

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))
print(model)
print("Testing tensor")
print(model:forward(torch.rand(512*2*2)))


if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(2), cudnn)
end

--load surrogate classifier
print(c.blue '==>' ..' loading classifier')
local surrogate_net = torch.load(opt.surrogate_path)

print(surrogate_net:get(14):get(2))

--load data
print(c.blue '==>' ..' loading data')
provider = torch.load '../dat/provider.t7' --load provider data
provider.trainData.data = provider.trainData.data:float() --convert to float
provider.valData.data = provider.valData.data:float()

--initialize confusion matrix
confusion = optim.ConfusionMatrix(opt.num_targets)

--set up logging
print('Will save at '..opt.save)
paths.mkdir(opt.save)
valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))
valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
valLogger.showPlot = false

parameters,gradParameters = model:getParameters()

-- Convolve function, for applying 32x32 pretraining model all over 96x96 image
function convolve(im_batch, net, kW, dW)
    --[[ Apply network convolutionally and get output from desired layer
    args:
        im_batch: batch of images to run through net
        net: network to run through
        kW: size of input to net
        dW: stride; how much to move window each iteration.
        
    --]]
    local output_size = 512 --TODO: replace with variable input
    local dW = dW or 1
    local n_steps = torch.floor((im_batch:size()[3]-kW)/dW+1)
    local batch_size = im_batch:size()[1]

    local output = torch.CudaTensor(batch_size,output_size,n_steps,n_steps)
    for i=1,n_steps,dW do
        for j=1,n_steps,dW do
            local patch = im_batch[{{},{},{1,kW},{1,kW}}]:contiguous():cuda()
            net:forward(patch)
            local last = net:get(14):get(2) --TODO: replace with variable input
            output[{{},{},i,j}] = last.output:cuda() --will result in batch_size x output_size vector at each window
        end
    end
    
    local pool_width = torch.floor(n_steps/2)
    local pooling = nn.SpatialMaxPooling(pool_width,pool_width):cuda()
    local pooled_output = pooling:forward(output)
    collectgarbage()
    return pooled_output
end

-- set criterion to SVM criterion
print(c.blue'==>' ..' setting criterion')
criterion = nn.MarginCriterion():cuda() --loss function


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size.  This removes the "remainder" when you divide the data by batch size
  indices[#indices] = nil

  local tic = torch.tic() --start timer
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices) -- progress bar is cool

    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    inputs = convolve(inputs,surrogate_net,opt.patch_size) -- transform using pretrained model

    local feval = function(x) --this is pretty much always the same for all torch programs
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets) --how well did you do
      local df_do = criterion:backward(outputs, targets) --get derivatives for every parameter
      model:backward(inputs, df_do)
      
      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic))) -- toc stops timer

  train_acc = confusion.totalValid * 100

  confusion:zero() -- reset to zero
  epoch = epoch + 1
end


function val()
  -- disable flips, dropouts and batch normalization -- what is batch normalization?
  model:evaluate()
  print(c.blue '==>'.." validating")
  local bs = 25
  for i=1,provider.valData.data:size(1),bs do
    local inputs = provider.valData.data:narrow(1,i,bs)
    inputs = convolve(inputs,surrogate_net,opt.patch_size)
    local outputs = model:forward(inputs)
    confusion:batchAdd(outputs, provider.valData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('val accuracy:', confusion.totalValid * 100) --get validation accuracy
  
  if valLogger then
    paths.mkdir(opt.save)
    valLogger:add{train_acc, confusion.totalValid * 100}
    valLogger:style{'-','-'}
    valLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/val.log.eps %s/val.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/val.png -out %s/val.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/val.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % 5 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  val()
end


