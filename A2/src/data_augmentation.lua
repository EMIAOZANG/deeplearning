require 'torch'
require 'image'
require 'nn'

function get_im_var(im)
    var=0
    for j=1,3 do
        var = var + torch.var(im[j])
    end
    return var
end

function augment_images(images, batch_num, num_transformations, patch_size)
    --[[
    takes batch of images, applies transformations, gives them labels, and returns features and labels
    args: images.  4D tensor of images
    returns:
        features: 4D tensor of images
        labels: 1D tensor of labels
    --]]
    local batch_size = images:size(1)
    local labels = torch.Tensor(batch_size * num_transformations)
    local features = torch.Tensor(batch_size * num_transformations,3,patch_size,patch_size)
    
    for i=1,batch_size do
        for j=1,num_transformations do
            local image = images[i]
            image = get_patch(image, 2*patch_size, 0.05)
            image = center_crop(image, patch_size)
            image = rotate(image, patch_size)
            image = shift_hue(image)
            labels[(i-1)*num_transformations + j] = (batch_num-1) * (batch_size-1) + i
            features[(i-1)*num_transformations + j] = image
        end
    end
    return features, labels
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

function shift_hue(im)
    -- shift hue by a random normal number mean 0 var 1/10
    local hue_shift = torch.div(torch.randn(1),10)
    im = image.rgb2hsl(im)
    im[1]:add(hue_shift[1])
    im = image.hsl2rgb(im)
    return im
end