--[[
   model_tuna.lua
   loop over some sets of params to get the best model, other params will remain default
]]

require 'xlua'

opt = lapp[[
   -f, --filename (default 'main.lua')
]]

-- params to loop through, you can add more to the table
local loop_params = {
   dropout = {0, 0.2, 0.5},
   layers = {2, 4, 6}
}


-- loop body, we loop each param only once to avoid too many runs
for key, value_table in pairs(loop_params) do
   -- Trains 1 epoch and gives validation set ~182 perplexity (CPU).
   print("Tuning params:\t"..key)
   for i = 1, #value_table do
      params = {
                batch_size=20, -- minibatch
                seq_length=20, -- unroll length
                layers=2,
                decay=2,
                rnn_size=200, -- hidden unit size
                dropout=0, 
                init_weight=0.1, -- random weight initialization limits
                lr=1, --learning rate
                vocab_size=10000, -- limit on the vocabulary size
                max_epoch=4,  -- when to start decaying learning rate
                max_max_epoch=15, -- final epoch
                max_grad_norm=5, -- clip when gradients exceed this norm value
                architecture = 'lstm',
                model_dir = './models/',
                result_path = './dat/exp_results'
      }
      params[key] = value_table[i] -- set key
      dofile(opt.filename)
      min_amortized_perp = nil
   end
end
