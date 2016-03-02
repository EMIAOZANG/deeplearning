--[[
data_augmentation.lua

given a tensor of images, augment each image several times.
returns a set of batches, where each batch is a table containing features and labels
features is a 4D tensor of images, and labels is a 1D tensor of ids for each image.
--]]


require 'torch'
require 'image'
require 'nn'
require 'xlua'
require 'optim'
local c = require 'trepl.colorize' --prints in color!


dofile('provider.lua')
print '==> processing options'

print(c.blue '==>' ..' loading data')
provider = torch.load '../dat/provider.t7' --load provider data
provider.trainData.data = provider.trainData.data:float() --convert to float
provider.valData.data = provider.valData.data:float()

opt = lapp[[
   --data                     (default "train")      data to augment
   -s,--save                  (default "../dat/augmented_images")      subdirectory to save logs
   -b,--batchSize             (default 64)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --num_transformations      (default 100)         number of trasnformations/augmentations per image
   --patch_size               (default 32)          size of patches to select from each image
]]

print(opt)


print('Will save at '..opt.save)
paths.mkdir(opt.save)

function get_im_var(im)
    var=0
    for j=1,3 do
        var = var + torch.var(im[j])
    end
    return var
end

function get_patch(im, patch_size, min_var)
    --[[ get a patch with variance more than 0.05
    args:
        im: input image
        patch_size: size of patch
        min_var: minimum variance
    
    returns: 
        im_crop: image of size patch_size
    --]]
    if im:size()[2]<patch_size then
        print("ERR: Patch size greater than image size")
        return
    end
    
    local var = 0
    while var<min_var do
        local max_x = im:size()[2] - patch_size
        local x1 = torch.round(torch.uniform(0,max_x))
        local x2 = x1 + patch_size
        local y1 = torch.round(torch.uniform(0,max_x))
        local y2 = y1 + patch_size
        im_crop = image.crop(im,x1,y1,x2,y2)
        var = get_im_var(im_crop)
    end
    return im_crop
end

function center_crop(im, patch_size)
    im_width = im:size()[2]
    if im:size()[2]<patch_size then
        print("ERR: Patch size greater than image size")
        return
    end
    -- center crop up to half of the image.
    local max_crop = (im_width - patch_size)/2
    local crop_margin = torch.round(torch.uniform(0,max_crop))
    im = image.crop(im, crop_margin, crop_margin, im_width - crop_margin, im_width - crop_margin)
    im = image.scale(im,patch_size, patch_size)
    return im
end

function rotate(im)
    -- rotate image a random amount between +/- 20 degrees and crop to fit
    width0 = im:size()[2]
    max_angle = 20/360*2*3.14 --20 degrees
    angle = torch.uniform(-max_angle,max_angle)
    local width1 = width0/(torch.abs(torch.sin(angle)) + torch.abs(torch.cos(angle)))
    local crop_margin = (width0 - width1)/2
    im = image.rotate(im,angle)
    im = image.crop(im,crop_margin,crop_margin,im:size()[2] - crop_margin,im:size()[3] - crop_margin)
    im = image.scale(im,width0,width0)
    return im
end

function process_rand()
    -- for producing random numbers in the appropriate range for the shift_hsl function
    vect = torch.rand(3)
    vect[1] = vect[1]*(4-.25) + .25 --the paper says to raise to a power between 0.25 and 4, but that gives me nulls
    vect[2] = vect[2]*(1.4-0.7) + 0.7
    vect[3] = vect[3]*(0.1+0.1) - 0.1
    return vect
end

function shift_hsl(im)
    -- shift hue by a random normal number mean 0 var 1/10
    --local hue_shift = torch.div(torch.randn(1),10)
    local hue_shift = torch.uniform(-0.1,0.1)
    local s_shift = process_rand()
    local l_shift = process_rand()
    im = image.rgb2hsl(im)

    --shift hue
    im[1]:add(hue_shift) 
    
    --shift saturation
    --im[2]:pow(s_shift[1]) --TEMP: killing this - doesn't work with normalization
    im[2]:mul(s_shift[2])
    im[2]:add(s_shift[3])

    --shift lightness
    --im[3]:pow(l_shift[1]) -- TEMP: killing this - doesn't work with normalization
    im[3]:mul(l_shift[2])
    im[3]:add(l_shift[3])
    im = image.hsl2rgb(im)
    return im
end

function augment_image_batch(image_batch, batch_num)
    --[[
    takes batch of images, applies transformations, gives them labels, and returns features and labels
    args: images.  4D tensor of images
    saves:
        features: 4D tensor of images
        labels: 1D tensor of labels
    --]]
    local batch_size = image_batch:size(1)
    local labels = torch.Tensor(batch_size * opt.num_transformations)
    local features = torch.Tensor(batch_size * opt.num_transformations,3,opt.patch_size,opt.patch_size)
    
    for i=1,batch_size do
        for j=1,opt.num_transformations do
            local image = image_batch[i]
            image = get_patch(image, 2*opt.patch_size, 0.05)
            image = center_crop(image, opt.patch_size)
            image = rotate(image, opt.patch_size)
            image = shift_hsl(image)
            labels[(i-1)*opt.num_transformations + j] = (batch_num-1) * (batch_size-1) + i
            features[(i-1)*opt.num_transformations + j] = image
        end
    end

    local data = {features=features, labels=labels}
    local filepath = opt.save.."/batch_"..batch_num..".t7"
    torch.save(filepath,data)
end

function augment_all_data(dataset)

    -- shuffle data and draw random batches
    local indices = torch.randperm(dataset:size()[1]):long():split(opt.batchSize)

    for batch_num,v in ipairs(indices) do
        print("Augmenting batch "..batch_num.."...")
        -- create batch
        batch = dataset:index(1,v)
        augment_image_batch(batch,batch_num)
    end
end

if opt.data=="train" then
    augment_all_data(provider.trainData.data)
elseif opt.data=="val" then
    augment_all_data(provider.valData.data)
else print("Ineligible dataset")
end


--[[
TODO:
0. Load 'train' or 'val' into memory, 
1. Load 'extra' into memory, shuffle, save 8000 to 'mini-extra.t7b'
2. Load 'mini-extra' into memory.  In batches of 100 at a time: augment batch, save to file
3. shuffle batch before training

Load data, augment, and return
--]]
