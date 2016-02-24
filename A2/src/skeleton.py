#data augmentation

def rotate(im):
    '''
    Jiayi:rotate image by random number from -20 to +20 degrees
    returns:
        rotated image
    '''

def scaling(im):
    '''
    Charlie:
    scale image by random factor between 0.7 and 1.4
    returns: scaled image
    '''

def translate(im):
    '''
    translate
    '''

def get_patch(raw_data):
    '''
    Charlie:
    1. pick a random image
    2. take patch of random size and location from original image
    3. rescale patch to 32x32
    4. return 32x32 patch, label
    TO DO LATER: get patches that have high gradient

    returns: patch rescaled to 32x32, label
    '''

def generate_data(num_classes, num_samples_per_class, raw_data):
    '''
    Jiayi:
    takes raw data and applies data augmentation transformations (translate, scale, rotate, etc)
    and outputs label and 
    '''

    for n in range(num_classes)
        raw_patch, label = get_patch(raw_data)
        
        for i in range(num_samples_per_class):    
            patch = scaling(raw_patch)
            patch = rotate(patch)
            patch = translate(patch)
            return patch

        #add patch to feature tensor
        #append_label_to_list of labels

    return features, labels

def train_model(features, labels):
    #TODO