-- Global params segment
UNKNOWN_KEY_ = '<unk>'

stringx = require('pl.stringx')
util = require('util')
require('base')
require 'xlua'
require 'io'
require 'nn'
require 'nngraph'

-- command line args
params = lapp[[
   -m, --model    (default "./models/lstm.net"),
   -g, --generation (default "multinomial"),
   -l, --layers (default 2),
   -d, --debug (default true),
   -b, --batch_size (default 20)
]]

-- IO and looping functions
function readline()
   --[[
      defines the io read behavior, throws exceptions when the input is EOF or invalid
      args:
         None
      returns:
         line: the line of input (a list with format [number, word1, word2, ...])
   ]]
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  local ext_length = nil

  if tonumber(line[1]) == nil then 
     error({code="init"})
  else
     ext_length = tonumber(table.remove(line, 1))
  end
  for i = 1,#line do
    if line[i] == nil then error({code="vocab", word = line[i]}) end
    line[i] = line[i]:lower() -- convert to lower case
  end
  if params.debug then print("readline processed table:",line) end
  return {ext_length, line}
end

function init_model(mfn)
   --[[
      Load and initialize the trained model, with output node enabled
      args:
         mfn : path to model file
      returns:
         m : initialized model
   ]]
   if mfn then
      local rnn_model = torch.load(mfn)
      g_disable_dropout(rnn_model.rnns)
      util.reset_state(rnn_model) -- btw why reset here?
      g_replace_table(rnn_model.s[0], rnn_model.start_s) -- initialize s[0] with start_s, which is all 0s
      return rnn_model -- returns the cofigured model
   end
   error({code="fnerr"}) --throw fnerr exception if model file is not loaded
   return nil
end

function sequence_generation(input_seq, ext_len, gen_mode)
   --[[
      use the loaded model to generate word sequence
      args:
         input_seq : input word sequence
         ext_len : the length of sequence that needs to be generated
         gen_mode : determines how the predicted word will be generated (multinomial or top)
      returns:
   ]]

   -- convert input sequence of words to indices
   local iter_length = #input_seq + ext_len -- number of iterations
   if params.debug then 
      print('input sequence:', input_seq) 
      print('iter_length: ', iter_length)
   end
   for i = 1, #input_seq do
      input_seq[i] = util.word2idx(input_seq[i], vocab_map) -- convert word to index
   end
   if params.debug then print('mapped input sequence: ', input_seq) end

   -- load and initialize RNN model, only one cell is needed, we will train the same cell repeatedly
   local m = init_model(params.model)
   local x = torch.Tensor()


   for i = 1, iter_length do
      x = util.fill_batch(input_seq[i], params.batch_size) -- create a batch input
      m_err, m.s[1], log_preds = unpack(m.rnns[1]:forward({x, x, m.s[0]})) -- use only the first RNN cell, proceed in iterative manner
      g_replace_table(m.s[0], m.s[1])

      probs = torch.exp(log_preds:select(1,1)) -- log_preds are batch_size * output_width, therefore we need to select only the first row to make it (width,)

      -- generate predicted id 
      local pred_id = inv_vocab_map[UNKNOWN_KEY_] -- generate <unk> if no method were specified
      if gen_mode == 'top' then
         pred_id = util.top_sample(probs)
      elseif gen_mode == 'multinomial' then
         pred_id = util.multinomial_sample(probs) 
      end
      if params.debug then print('pred_id: ', pred_id) end

      -- insert the predicted word back to the input sequence if i > #input_seq (initial)
      if i >= iter_length - ext_len then
         input_seq[#input_seq + 1] = pred_id
      end
   end
   g_enable_dropout(m.rnns)


   -- convert id to words and return the table
   local output = {}
   for i = 1, iter_length do
      output[i] = util.idx2word(input_seq[i], inv_vocab_map)
   end
   return output
end

-- main
vocab_map = torch.load('./dat/v_map.t7b')
print("Dictionary loaded")
inv_vocab_map = util.inverse_mapping(vocab_map)
torch.save('./dat/inv_map.t7b', inv_vocab_map)
print("Inverse Mapping Created and Saved in ./dat, size: "..#inv_vocab_map)

while true do
  print("Query: len word1 word2 etc")
  local ok, line = pcall(readline) -- call function in protected mode, returns the (exit_state_flag, line_content)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
   --process valid input
    local ext_length = line[1]
    local sequence = line[2] -- it's a table of strings

    -- generate new sequence using trained language model
    new_sequence = sequence_generation(sequence, ext_length, params.generation)

    -- output to screen
    for i = 1, #new_sequence do
       io.write(new_sequence[i]..' ') --use io.write to avoid linebreak
    end
    print('') 
  end
end
