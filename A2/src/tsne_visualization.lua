require 'nn'
require 'image'
mm = require 'manifold';

--definition of tSNE visualization
function tSNEVis(imgs, layerOutput, imSize, pcaDim)
   --[[
      perform t-SNE visualization given image dataset and the feature dataset 
      args:
         imgs: the image dataset, 
         layerOutput: output of some layer in the prediction model that will be used for clustering
         imSize: the sisze of tSNE visualization image
         pcaDim: dimension of PCA for preprocessing, must be smaller than the rank of imgs
      return:
         the tSNE image, torch.DoubleTensor(3,imSize,imSize)
   ]]
   local lout = torch.DoubleTensor(layerOutput:size()):copy(layerOutput:double())
   --print('Layer output shape: '..(lout:size()))
   lout:resize(lout:size(1),lout:size(2)*lout:size(3)*lout:size(4))
   --print('Flattened layer output shapei: '..lout:size())
   print(lout:size())

   local opts = {ndims = 2, perplexity = 30, pca = pcaDim, use_bh = true, theta = 0.5}
   mapped_lout = mm.embedding.tsne(lout, opts)

   --print('tSNE completed, reduced the dataset to:'..mapped_lout:size())
   
   
   --save a float copy of img data
   local x = torch.DoubleTensor(imgs:size()):copy(imgs:double())
   --print('Image Data Size:'..x:size())
   
   map_im = mm.draw_image_map(mapped_lout, x, imSize, 0, true)
   collectgarbage()

   return map_im
end
