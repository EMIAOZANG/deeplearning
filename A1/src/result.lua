-- result.lua

-- given a model, saves a list of predicted values on the MNIST test set.

-- NOTE: This assumes that it shares a directory with both model.net and 1_data.lua

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars

opt = {
  size="full", -- size of data
  visualize=false, -- don't enable visualizations
  sub=true, -- indicates that this is the submission run
  }

function predict(model_path)
  -- loads model and runs it on test data to generate 1D tensor of predictions

   model = torch.load(model_path)

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- run through test data
   print('==> getting predictions from test set:')
    
   -- Initalize prediction tensor
   local preds = torch.Tensor(testData:size())
    
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]:double()

      -- get value and index of the max predicted value. 
      local m,pred = torch.max(model:forward(input),1)

      -- append t,pred to csv
      preds[t]=pred
   end
    
   return preds
end --predict

function save_pred_file(fname, preds)
  -- save list of predictions to file fname

  --[[
  args:
    fname: file name of save file
    preds: tensor of predictions
  returns:
    none. Saves to disk
  ]]

    local file = io.open(fname,"w") --create file
    file:write("Id,Prediction\n") --write header
    io.close(file)
    
    local file = io.open(fname,"a") -- put file in append mode

    --loop through preds list and save each line to file.
    for t = 1,preds:size()[1] do
        file:write(t .. ',' .. preds[t] .. "\n")
    end
    io.close(file)
end

-- RUN
dofile '1_data.lua' -- run to get testData
predictions = predict("model.net")
save_pred_file("predictions.csv",predictions)
