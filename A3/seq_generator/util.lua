stringx = require('pl.stringx')
ptb = require('data')
require 'xlua'
require 'io'

local function word2idx(word, v_map)
   --[[
      get index number corresponding to the word in v_map, returns the index of <unk> if the word is not in the dictionary 
      args:
         v_map : word mapping dictionary 
   ]]
   local idx = v_map[word]
   if idx == nil then
      idx = v_map[UNKNOWN_KEY_] -- if the word is not in the dictionary, treat it as unknown
   end
   return idx
   
end

local function idx2word(idx, inv_map)
   --[[
      find the word corresponding to the index value in inv_map and returns the string
      args:
         idx : index value in inverse map
         inv_map : the inverse map 
      returns:
         the word found or empty string if the idx value is not found
   ]]
   return true and inv_map[idx] or ""
end

local function fill_batch(idx, batch_size)
   --[[
      fill an input batch with a single word index, for sequence generation purpose
      args:
         idx: word index
         batch_size: batch length used during training phase
   ]]
   return torch.Tensor(batch_size):fill(idx)
end

local function reset_state(model)
   --[[
      reset model start_state, start_state is required to be non-nil
      args:
         model : the RNN model to be reset
   ]]
   if model ~= nil and model.start_s ~= nil then 
      for d = 1, 2 * params.layers do
         model.start_s[d]:zero()
      end
   end
end

local function inverse_mapping()
   --[[
      create a inverse indexing for vocab_map
   ]]
    for w, i in pairs(vocab_map) do
       inv_vocab_map[i] = w
    end
end

-- encapsulate member functions in a Table to avoid global namespace pollution
return {
   word2idx = word2idx,
   idx2word = idx2word,
   fill_batch = fill_batch,
   reset_state = reset_state,
   inverse_mapping = inverse_mapping
}
