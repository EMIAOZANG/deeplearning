require 'nn'
require 'class'
require 'cunn'
require 'image'

do --data augmentation module and class definition
   local BatchRotate, parent = class('nn.BatchRotate', 'nn.Module')

   function BatchRotate:__init()
      parent.__init(self)
      self.train = true
   end

   function BatchRotate:updateOutput(input)
      if self.train then
         local bs = input:size(1) --get the size of 1st dimension (number of images)
         local rotate_mask = torch.randperm(bs):le(bs/2)
         for i=1,input:size(1) do
            if rotate_mask[i] == 1 then image.rotate(input[i], input[i],0.35*2*(math.random()-0.5)) end --performs local rotation, copy rotation probably better?
         end
         self.output:set(input)
      end
   end

end



