#Plan of Attack

##Labeled Baseline
1. train model only on the labeled data; measure performance
1. submit to practice submission script
1. implement flipping, scaling and translating data augmentation


##Unlabeled Pre-Training
Wherein we take advantage of the unlabeled data

### Universum Prescription Approach
1. give unlabeled data a 'dustbin' label, as in the Universum Prescription paper
1. train standard convolutional net using this approach
1. compare results to baseline
1. submit so we have something that beats baseline

###Denoising auto-encoder/self-classifier.
1. Pretraining layer for feature extraction. For each starting image, augment unlabeled data (flip, scale, rotate, translate) and build model to map augmented images to label identifying starting image.  For example if we start with two images, "A" and "B", augment them so that we have ~8 images for each starting image and build a classifier to classify the augmented images as either "A" or "B". The last hidden layer of this classifier will be our features for step 2. Train this on labeled data alone, unlabeled data, and both. 
2. Use the pretraining layer to extract features from the labeled images.  Build SVM model or fully-connected MLP to classify using these extracted features. 
3. Submit

###Convolutional Clustering
1. Read paper