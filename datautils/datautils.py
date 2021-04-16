#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:25:52 2019

@author: robert
"""

import os
import numpy as np
import cv2
import magic

def isimage(filename):
    mimetype = magic.from_file(filename, mime=True)
    res = mimetype.split('/')[0]
    if res == 'image':
        return True
    else:
        return False

def slice_frames(source, step):
    '''
    Slices video to frames taking every n'th frame, where n = step.
    
    Syntax:
        
        slice_frames(source, step)
        
    Parameters:
        
        source (src): path to video file.
        
        step (int):
            
    '''
    source_dir = os.path.join(*os.path.split(source)[:-1])
    target_dir = os.path.split(source)[-1].split('.')[0]
    target_dir = os.path.join(source_dir, target_dir)
    os.mkdir(target_dir)
    cap = cv2.VideoCapture(source)
    n_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if n_frame % step == 0:
                cv2.imwrite(os.path.join(target_dir, str(n_frame)+'.png'),
                            frame)
            n_frame += 1
        else:
            break
    cap.release()
                                                
def annotate(root, key_class_dict=None):
    '''
    Tool for annotating images. Requires OpenCV.
    Loops through images in root directory. Press a key to move current image into directory defined by key_class_dict, or "Esc" to break.

    Syntax:
        
        annotate(root, key_class_dict)
        
    Parameters:
        
        root (str): directory with image files.
        
        key_class_dict (dict): dict of format {key_character:directory_name}, e.g. {'0':'cats', '1':'dogs'}.
                               If None, first characters of existing directory names are used (press "c" to move current image to "root/cats/"). 

    '''
    listdir = os.listdir(root)
    files = filter(lambda s:os.path.isfile(os.path.join(root, s)), listdir) 
    dirs = list(filter(lambda s:os.path.isdir(os.path.join(root, s)), listdir)) 
    first_chars = list( map (lambda s:s[0], dirs))
    files = map(lambda s:os.path.join(root, s), files ) 
    dirs = sorted( list( map(lambda s:os.path.join(root, s), dirs ) ))
    files = list( filter (isimage, files) )
    if key_class_dict is None:
        coding = {ord(first_chars[i]):dirs[i] for i in range(len(dirs))}
    else:
        coding = {ord(key):key_class_dict[key]\
                       for key in key_class_dict}
    print('Press key:\tTo move image into folder:')
    for c in coding:
        print(chr(c),'\t\t',coding[c])
    print('Press any other key to skip image.')
    print('Press ESC to quit.')
    inp = ''
    while not inp in ('n','N','y','Y'):
        inp = input('Ready to start (y/n)?')
    if inp in ('n','N'):
        return
    for path in files:
        img = cv2.imread(path)
        cv2.imshow('_',img)
        key = cv2.waitKey()
        if key & 0xFF == 27:
            break
        try:
            target_class = coding[key]
        except KeyError:
            continue
        target_dir = os.path.join(root, target_class)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        cv2.imwrite(os.path.join(target_dir,
                                 os.path.split(path)[-1]), img)
        os.remove(path)
    cv2.destroyAllWindows()
   
def annotateROI(root, 
                key_class_dict=None):
    '''
    Tool for selecting and annotating a region of interest (ROI) on an image.

    Syntax:
        
        annotate(root, key_class_dict)
        
    Parameters:
        
        root (str): directory with image files.
        
        key_class_dict (dict): dict of format {key_character:directory_name}, e.g. {'0':'cats', '1':'dogs'}.
                               If None, first characters of existing directory names are used (press "c" to move current image to "root/cats/"). 

    '''
    listdir = os.listdir(root)
    files = filter(lambda s:os.path.isfile(os.path.join(root, s)), listdir) 
    dirs = list(filter(lambda s:os.path.isdir(os.path.join(root, s)), listdir)) 
    first_chars = list( map (lambda s:s[0], dirs))
    files = map(lambda s:os.path.join(root, s), files ) 
    dirs = sorted( list( map(lambda s:os.path.join(root, s), dirs ) ))
    files = list( filter (isimage, files) )
    if key_class_dict is None:
        coding = {ord(first_chars[i]):dirs[i] for i in range(len(dirs))}
    else:
        coding = {ord(key):key_class_dict[key]\
                       for key in key_class_dict}
    print('Press key:\tTo move image into folder:')
    for c in coding:
        print(chr(c),'\t\t',coding[c])
    print('Press any other key to skip image.')
    print('Press ESC to quit.')
    inp = ''
    while not inp in ('n','N','y','Y'):
        inp = input('Ready to start (y/n)?')
    if inp in ('n','N'):
        return
    for path in files:
        img = cv2.imread(path)
        [h, w] = img.shape[:2]
        cv2.imshow('_',img)
        filename = os.path.split(path)[-1]
        name = '.'.join(filename.split('.')[:-1])
        labelname = name+'.txt'
        labelpath = os.path.join(root,'labels',labelname)
        roi = cv2.selectROI('_', img, False)
        key = cv2.waitKey()
        if key & 0xFF == 27:
            break
        try:
            target_class = coding[key]
        except KeyError:
            continue
        roistr = f'{target_class} {roi[0]/w} {roi[1]/h} {(roi[2])/w} {(roi[3])/h}'
        print(roistr, file=open(labelpath, 'w'))
        target_dir = os.path.join(root, 'images')
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        cv2.imwrite(os.path.join(target_dir,
                                 filename), img)
        os.remove(path)
    cv2.destroyAllWindows()

class ImageLoader():
    '''
    Loader for image dataset contained in a file structure:

        root--class1--image1
           |       |--image2
           |
           |--class2--image3
           
    Requires OpenCV. Subfolders inside "class" folders are ignored. If there is no "class" level, label 0  is assigned to all images.
    
    Syntax:
        
       loader = ImageLoader(root, class_labels)
           
    Parameters:
           
       root (str): root directory name
       
       class_labels (dict): dict of the kind {'class1_name':class1_label, ...}
       If None, labels are assigned to classes in lexicographical order.
       
    Method:
        
       load()
       
    '''
    def __init__(self, root, 
             class_labels = None): 
        self.root = root
        if class_labels is None:
            class_names = [d for d in os.listdir(root)\
                           if os.path.isdir(os.path.join(root,d))]
            class_names = sorted(class_names)
            self.class_labels = {class_names[i]:i for i in range(len(class_names))}
        else:
            if not type(class_labels) is dict:
                raise TypeError('Class_labels parameter must be dict {class_name:class_label}')
            self.class_labels = class_labels
            
    def load(self,
             shuffle = False,
             grayscale = False,
             normalize = False,
             flatten = False,
             check_equal_size = True):
        '''
        Loads the dataset.
        
        Syntax:
            
            X, y = loader.load(shuffle, grayscale, normalize, flatten, check_equal_size)
    
        Parameters:
            
            shuffle (bool): set True to shuffle data. Default value: False
            
            grayscale (bool): set True to convert images to grayscale. Default value: False           
            
            normalize (bool): set True to apply [0..255) -> [0, 1) mapping to images. Default value: False
        
            flatten (bool): set True to reshape each image into [1, height*width*n_channels] array. Default value: False
            
            check_equal_size (bool): if True and images are not the same size, raises ValueError. Default value: False
    
        Returns:
            
            data (ndarray): array of images
                            If flatten=False, data shape is (n_samples, image_height, image_width) for grayscale images or (n_samples, image_height, image_width, 3) for BGR images.
                            If flatten=True, data shape is (n_samples, image_height*image_width*n_channels).
    
            labels (ndarray): array of class labels of shape (n_samples,)          
            
        '''
        
        def load_image_list(folder):
            X = []
            for file_name in os.listdir(folder):
                path = os.path.join(folder,file_name)
                if os.path.isfile(path) and isimage(path):
                    if grayscale:
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    else:
                        img = cv2.imread(path)
                    X.append(img)
            return X

        X, y, shapes = [], [], []
        if len(self.class_labels) > 0: #labelled data
            for class_name in self.class_labels:
                class_dir = os.path.join(self.root,class_name)
                if os.path.exists(class_dir) and os.path.isdir(class_dir):
                    X_add = load_image_list(class_dir)
                    y_add = [self.class_labels[class_name]]*len(X_add)
                    shapes_add = [x.shape for x in X_add]
                    X += X_add
                    y += y_add
                    shapes += shapes_add
            shapes = set(shapes)
        else: #unlabelled data
            X = load_image_list(self.root, grayscale)
            y = [0]*len(X)
            shapes = {x.shape for x in X}
        if len(X) == 0: #try to load unlabelled data
            raise ValueError('No images loaded.')
        if check_equal_size and len(shapes) > 1:
            raise ValueError('All images must be the same size.')
        X, y = np.array(X), np.array(y)
        if shuffle:
            idx = np.random.permutation(len(y))
            X, y = X[idx], y[idx]
        if normalize:
            X = X.astype(float)/255.
        if flatten:
            old_shape = X[0].shape
            new_shape = old_shape[0]*old_shape[1]
            if len(old_shape)==3:
                new_shape *= old_shape[2]
            X = X.reshape(-1, new_shape)
        return X, y
    
