--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
--
--TODO:
--1. make main.lua run 
--2. implement gru(x, prev_c, prev_h)
--3. modify fp and bp so they can work with GRU and LSTM simultaneously
----

gpu = false
if gpu then
    require 'cunn'
    print("Running on GPU") 
    
else
    require 'nn'
    print("Running on CPU")
end

require('nngraph')
require('base')
ptb = require('data') --import user lib data.lua

-- Trains 1 epoch and gives validation set ~182 perplexity (CPU).
if params == nil then -- find params in the outer loop first
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
                max_max_epoch=5, -- final epoch
                max_grad_norm=5, -- clip when gradients exceed this norm value
                architecture = 'lstm',
                model_dir = './models/',
                result_path = './dat/exp_results.txt'
               }
end

args_concat_string = 'params='..(params.architecture)..'_'..tostring(params.dropout)..'_'..tostring(params.layers)
print(args_concat_string)
-------------------------------------------------------
function transfer_data(x)
   --[[
      transfers the data to appropriate type (cuda or normal)
   ]]
    if gpu then
        return x:cuda()
    else
        return x
    end
end

--[[
   model is a table that provides abstraction of a RNN LSTM network
   Members:
   s : state, each state is a table, which carries all data in RNN blocks
   ds : delta_s
   start_s : starting state
   rnns : a sequence of RNN blocks
   err : errors, initalized as a 0 vector

]]
model = {}

local function lstm(x, prev_c, prev_h)
   --[[
      creates LSTM cell module
      args:
         x : input
         prev_c : C[t-1]
         prev_h : h[t-1]
      returns:
         next_c : C[t]
         next_h : h[t]
   ]]
    -- Calculate all four gates in one go
    local i2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
    local h2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
    local gates            = nn.CAddTable()({i2h, h2h})

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates   =  nn.Reshape(4,params.rnn_size)(gates)
    local sliced_gates     = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_c, next_h
end

local function gru(x, prev_c, prev_h)
   --[[
      creates GRU cell module
   ]]
end

function create_network()
    local x                  = nn.Identity()() -- input batch
    local y                  = nn.Identity()() -- output batch?
    local prev_s             = nn.Identity()()
    local i                  = {[0] = nn.LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)} -- i is a (batchs_size, 200) vector
    local next_s             = {}
    local split              = {prev_s:split(2 * params.layers)} --splits s equally to two 2*layers table
    for layer_idx = 1, params.layers do
        local prev_c         = split[2 * layer_idx - 1]
        local prev_h         = split[2 * layer_idx]
        local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
        local next_c, next_h = lstm(dropped, prev_c, prev_h)
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h
    end
    local h2y                = nn.Linear(params.rnn_size, params.vocab_size)
    local dropped            = nn.Dropout(params.dropout)(i[params.layers])
    local pred               = nn.LogSoftMax()(h2y(dropped)) -- pred is a (vocab_size, ) log probability vector
    local err                = nn.ClassNLLCriterion()({pred, y}) -- 1 scalar for each time step (RNN block)
    local module             = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s), pred})
    -- initialize weights
    module:getParameters():uniform(-params.init_weight, params.init_weight) -- initialize weight with a uniform distribution U[-init_weight, init_weight]
    return transfer_data(module)
end

function setup(mode)
   --[[
   Create and initialize a RNN LSTM or GRU network
   ]]
   if mode == 'GRU' then
    print("Creating a RNN GRU network")
   else
    print("Creating a RNN LSTM network.")
    local core_network = create_network()
    paramx, paramdx = core_network:getParameters()
    model.s = {}
    model.ds = {}
    model.start_s = {}
    for j = 0, params.seq_length do
        model.s[j] = {}
        for d = 1, 2 * params.layers do
            model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        end
    end
    for d = 1, 2 * params.layers do
        model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.err = transfer_data(torch.zeros(params.seq_length))
   end
end

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

function fp(state)
    -- g_replace_table(from, to).  
    g_replace_table(model.s[0], model.start_s)
    
    -- reset state when we are done with one full epoch
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(state)
    end
    
    -- forward prop
    for i = 1, params.seq_length do
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        model.err[i], model.s[i], pred = unpack(model.rnns[i]:forward({x, y, s}))
        state.pos = state.pos + 1
    end
    
    -- next-forward-prop start state is current-forward-prop's last state
    g_replace_table(model.start_s, model.s[params.seq_length])
    
    -- cross entropy error
    return model.err:mean()
end

function bp(state)
    -- start on a clean slate. Backprop over time for params.seq_length.
    paramdx:zero()
    reset_ds()
    for i = params.seq_length, 1, -1 do
        -- to make the following code look almost like fp
        state.pos = state.pos - 1
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        -- Why 1?
        local derr = transfer_data(torch.ones(1))
        -- adding dpred so that pred could backprop as wellAnna Choromanska
        local dpred = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
        -- tmp stores the ds
        local tmp = model.rnns[i]:backward({x, y, s},
                                           {derr, model.ds, dpred})[3]
        -- remember (to, from)
        g_replace_table(model.ds, tmp)
    end
    
    -- undo changes due to changing position in bp
    state.pos = state.pos + params.seq_length
    
    -- gradient clipping
    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / model.norm_dw
        paramdx:mul(shrink_factor)
    end
    
    -- gradient descent step
    paramx:add(paramdx:mul(-params.lr))
end

function write_result(run_data, epoch, metric)
   local fp = io.open(params.result_path, 'a+')
   fp:write(args_concat_string..':\t'..run_data..'\t'..tostring(epoch)..'\t'..metric..'\n')
   fp:close()
end

function run_valid()
    -- again start with a clean slate
    reset_state(state_valid)
    
    -- no dropout in testing/validating
    g_disable_dropout(model.rnns)
    
    -- collect perplexity over the whole validation set
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(state_valid)
    end

    local amortized_perp = torch.exp(perp / len) -- calculate amortized perplexity of words

    if min_amortized_perp then
       if amortized_perp < min_amortized_perp then
         min_amortized_perp = min_amortized_perp -- update best result
         torch.save((params.model_dir)..args_concat_string..'_best.net', model)  
         print("Current best model saved to file")
      end
    else
       min_amortized_perp = amortized_perp
       torch.save((params.model_dir)..args_concat_string..'_best.net', model)  
       print("Current best model saved to file")


    end

    print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
    write_result("validation", epoch, g_f3(torch.exp(perp / len))) -- writing result to file
    g_enable_dropout(model.rnns)
    
end

function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns) -- could probably disable output node as well?
    local perp = 0
    local len = state_test.data:size(1)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    write_result("test", epoch, g_f3(torch.exp(perp/ (len -1)))) -- writing result to file
    g_enable_dropout(model.rnns)
end

---------------------
-- MAIN LOOP
---------------------

if gpu then
    g_init_gpu(arg)
end

-- get data in batches
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

print("Network parameters:")
print(params)

local states = {state_train, state_valid, state_test}
for _, state in pairs(states) do
    reset_state(state)
end
setup()
step = 0
epoch = 0
total_cases = 0
beginning_time = torch.tic()
start_time = torch.tic()
print("Starting training.")
words_per_step = params.seq_length * params.batch_size
epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)

while epoch < params.max_max_epoch do

    -- take one step forward
    perp = fp(state_train)
    if perps == nil then
        perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    
    -- gradient over the step
    bp(state_train)
    
    -- words_per_step covered in one step
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    
    -- display details at some interval
    if step % torch.round(epoch_size / 10) == 10 then
        wps = torch.floor(total_cases / torch.toc(start_time))
        since_beginning = g_d(torch.toc(beginning_time) / 60)
        print('epoch = ' .. g_f3(epoch) ..
             ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
             ', wps = ' .. wps ..
             ', dw:norm() = ' .. g_f3(model.norm_dw) ..
             ', lr = ' ..  g_f3(params.lr) ..
             ', since beginning = ' .. since_beginning .. ' mins.')
    end
    
    -- run when epoch done
    if step % epoch_size == 0 then
        run_valid()
        if epoch > params.max_epoch then
            params.lr = params.lr / params.decay
        end
    end

   -- stop training if perplexity does not improve any more: Optional
    
end

-- save model to file
model_path = './models/'..args_concat_string..'_final.net'
torch.save(model_path, model)
print("Model saved to ./models/")

--run test
run_test()
print("Training is over.")