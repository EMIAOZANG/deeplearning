require 'image'
require 'nn'

do
   local BatchGetPatch, parent = torch.class('nn.BatchGetPatch', 'nn.Module')

   function BatchGetPatch:__init()
      parent.__init(self)
      self.train = true
   end

   function BatchGetPatch:updateOutput(input, patchSize)
      if self.train then
         local bs = input:size(1)
         for i=1,bs do
            local x1_max = input[i]:size(3)-2*patchSize --patchSize is the patch size
            local y1_max = input[i]:size(2)-2*patchSize
            --get a random point from [1,x1_max] and [1,y1_max] as the topleft starting point of cropping
            local x1 = math.random(x1_max)
            local y1 = math.random(y1_max)
            local output = torch.Tensor(bs,3,2*patchSize)

            image.crop(input[i], input[i], x1,y1, x1+2*patchSize, y1+2*patchSize) --Crop each image in current batch
         end
      end

      self.output:set(output)
      return self.output
   end
end
