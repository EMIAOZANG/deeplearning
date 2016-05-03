--[[
   unittest_main.lua
   provides unit test for functions in main.lua that could run on small test cases to help debugging

   dependencies:
      luaunit
]]

require('luaunit')
require('main')

-------------
-- test class
------------- 
local TestGlobalFunctions = {}
   function setUp()
      self.params = {
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
                max_max_epoch=13, -- final epoch
                max_grad_norm=5, -- clip when gradients exceed this norm value
                architecture = 'lstm',
                model_dir = './models/',
                result_path = './dat/exp_results.txt'
               }
   end

   local function TestGlobalFunctions:test_write_result()
      local params = {
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
                max_max_epoch=13, -- final epoch
                max_grad_norm=5, -- clip when gradients exceed this norm value
                architecture = 'lstm',
                model_dir = './models/',
                result_path = './dat/exp_results.txt'
               }

      local run_data_ = "validation"
      local epoch_ = 0.904
      local metric = "182.34"
   end
-----------------
-- end test class
-----------------
LuaUnit:run()
