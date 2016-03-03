require 'xlua'
require 'optim'
require 'cunn'
require 'image'
local c = require 'trepl.colorize' --prints in color!

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 64)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default surrogate_classifier)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --imageDir                 (default '../dat/augmented_images')  directory of augmented image data
   --val_pct                  (default 0.1)           fraction of data to devote to validation
   --num_targets	      (default 4000)	     number of surrogate classes
]]

print(opt)

do -- data augmentation module -- local block of variables that will get killed
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module') -- extends nn.Module class, makes it usable as a layer in the nn.Sequential call

  function BatchFlip:__init() -- modify this to add rotation and translation to flip
    parent.__init(self)
    self.train = true --set train to true.  Only flip training data.
  end

  function BatchFlip:updateOutput(input)
    if self.train then -- is true upon init
      local bs = input:size(1) -- list of images (1st dimension is image id's)
      local flip_mask = torch.randperm(bs):le(bs/2) -- random list of 1s and 0s, so randomly flips some images.  different for each epoch
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end --performs a horizontal flip.  could add in vertical flip
      end
    end
    self.output:set(input)
    return self.output
  end
end

print(c.blue '==>' ..' configuring model')
dofile('models/'..opt.model..'.lua')
vgg = build_surrogate_classifier(opt.num_targets)

local model = nn.Sequential()
model:add(nn.BatchFlip():float()) -- call batch flip.  can add another rotation layer or translation if you like
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda()) -- model shift to 'cuda' mode
model:add(vgg:cuda()) --load model from external file
model:get(2).updateGradInput = function(input) return end -- get layer 2 of the model (batchflip).  take this input and drop it on the floor.  won't do anything in the backprop stage

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end

print(model)

confusion = optim.ConfusionMatrix(opt.num_targets)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))
valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
valLogger.showPlot = false

parameters,gradParameters = model:getParameters()

print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda() --loss function


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

function train_val_split(data,val_pct)
    --[[
    Split data into test and training sets, according to validation percent
    
    returns:
        both: table containing trainData and valData
    ]]
  local size = data.labels:size()[1]
  local shuffle = torch.randperm(size):long()
  local val_samples = torch.round(val_pct*size)
  local train_samples = size - val_samples
  local train_idx = shuffle[{{1,train_samples}}]
  local val_idx = shuffle[{{train_samples+1,size}}]
      
  local trainData = {}
  trainData['data']=data.features:index(1,train_idx)
  trainData['labels']=data.labels:index(1,train_idx)

  local valData = {}
  valData['data']=data.features:index(1,val_idx)
  valData['labels']=data.labels:index(1,val_idx)
    
  both = {trainData=trainData,valData=valData}
  return both
  trainData = nil
  valData = nil
  collectgarbage()
end

function load_data(fname)
  --load batch
  print(c.blue '==>' ..' loading '..fname..'...')
  local data = torch.load(opt.imageDir..'/'..fname)
  data.features = data.features:float()
  data.labels = data.labels:float()

  -- train val split
  batch = train_val_split(data,opt.val_pct)
  batch.trainData.data = batch.trainData.data:float() --convert to float
  batch.valData.data = batch.valData.data:float()
end


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(batch.trainData.data:size(1)):long():split(opt.batchSize) --get indices to split data into minibatches
  -- remove last element so that all the minibatches have equal size.  This removes the "remainder" when you divide the data by batch size
  indices[#indices] = nil

  local tic = torch.tic() --start timer
  for t,v in ipairs(indices) do -- iterate through mini-batches
    xlua.progress(t, #indices) -- progress bar is cool
    
    local inputs = batch.trainData.data:index(1,v)
    targets:copy(batch.trainData.labels:index(1,v))

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
  print(c.blue '==>'.." valing")
  local bs = 25
  for i=1,batch.valData.data:size(1),bs do
    local outputs = model:forward(batch.valData.data:narrow(1,i,bs))
--print(outputs)
--print(torch.max(outputs,2))
    confusion:batchAdd(outputs, batch.valData.labels:narrow(1,i,bs))
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


function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local pfile = popen('ls "'..directory..'"')
    for filename in pfile:lines() do
        i = i + 1
        t[i] = filename
    end
    pfile:close()
    return t
end

file_list = scandir(opt.imageDir)

for i=1,opt.max_epoch do
  for k,file in pairs(file_list) do
    load_data(file)
    train()
    val()
    batch = nil
    collectgarbage()
  end
end

