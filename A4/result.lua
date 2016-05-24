--[[
   result.lua
   loads the best model and runs the model on test data, output test perplexity to stdout
   default best model path = ./models/best_model.net
]]

require 'nn'
require 'nngraph'
require 'io'
require 'xlua'
require('base')
ptb = require('data')

opt = lapp[[
   -m, --model    (default './models/best_model.net'),
   -o, --output   (default ''),
   -b, --batchsize   (default 20)
   -l, --layers (default 2)
]]

local function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * opt.layers do
            model.start_s[d]:zero()
        end
    end
end

local function run_test()
   --[[
      counter part of the run_test function in main.lua which runs the model over test set and prints the test perplexity
      args:
         None
      returns:
         None, prints message to stdout of file
   ]]
    reset_state(state_test)
    print("Successfully reset states")
    g_disable_dropout(model.rnns) -- could probably disable output node as well?
    local perp = 0
    local len = state_test.data:size(1)
    print(len)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do

        xlua.progress(i, len - 1) -- progress bar is cool
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end


-- load the model file
if not paths.filep(opt.model) then
   if not paths.dirp('./models') then
      os.execute('mkdir models')
   end
   os.execute('wget https://www.dropbox.com/s/9eq88jzwnd755pr/best_model.net?dl=0 -O ./models/best_model.net')
end
model = torch.load(opt.model)
print("Model file loaded")

state_train = {data=ptb.traindataset(opt.batchsize)} -- build vocab_map on top of train set
state_test = {data=ptb.testdataset(opt.batchsize)}
run_test()
print("Program terminated")



