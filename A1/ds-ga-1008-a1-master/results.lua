-- results.lua
--[[
args:
	model: model saved in training 
	test data: dataset

returns:
	saves a list of predicted values
]]

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

opt = {size="full",visualize=false}
dofile '../ds-ga-1008-a1-master/1_data.lua'

-- get model and test data from command line?
function predict(model_path)
   --model path: "results/model.net"
   model = torch.load(model_path)

   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- run through test data
   print('==> getting predictions from test set:')
    
   -- Initalize prediction tensor
   local preds = torch.zeros(testData:size())
    
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      --if opt.type == 'double' then input = input:double()
      --elseif opt.type == 'cuda' then input = input:cuda() end
      --TEMP:
      input = input:double()
      --end TEMP
      local target = testData.labels[t]

      -- get value and index of the max predicted value. Index 10 represents 0
      local m,pred = torch.max(model:forward(input),1)

      -- append t,pred to csv
      preds[t]=pred
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to get predictions = " .. (time*1000) .. 'ms')
    
   return preds
end --predict

function save_pred_file(fname, preds)
    local file = io.open(fname,"w")
    file:write("Id,Prediction\n")
    io.close(file)
    
    local file = io.open(fname,"a")
    for t = 1,preds:size()[1] do
        file:write(t .. ',' .. preds[t] .. "\n")
    end
    io.close(file)
end

-- RUN
predictions = predict("results/model.net")
save_pred_file("results/predictions.csv",predictions)