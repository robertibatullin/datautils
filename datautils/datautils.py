#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:25:52 2019

@author: robert
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import magic

rot_angles = {90:Image.ROTATE_90,
              180:Image.ROTATE_180,
              270:Image.ROTATE_270}

flip_modes = {'hor':Image.FLIP_LEFT_RIGHT,
              'vert':Image.FLIP_TOP_BOTTOM}

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
    
class Tiling():
    '''
    Tool for slicing an image into square tiles of specified size. Requires PIL.
    
    Syntax:
        
        t = Tiling(size, stride)
        
    Parameterers:        
        
        size (int): tile width, equal to tile height.
        
        stride (int): tiling stride (by default stride = size).
        
    Methods:
        
        apply(source): applying the tiling to an image.
        
        get_tile_rects(): yields iterable of (left, top, right, bottom) tuples for every tile.

        get_tile_images(): yields iterable of tile images.
        
        write_tiles(target_dir): writes tiles as separate images into target_dir (by default "tiles" subfolder in source image folder).
        
    '''
    
    def __init__(self, size, stride=None):
        self.size = size
        if stride is None or stride < 1:
            self.stride = size
        else:
            self.stride = stride
            
    def apply(self, source):
        '''
        Applying the tiling to an image.
        
        Syntax:
            
            t.apply(source)
            
        Parameter:
            
            source (str or PIL Image): image to tile. Can be filename or PIL image object.
        '''

        if type(source) is str:
            self.source = Image.open(source)
            self.source_name = os.path.split(source)[-1].split('.')[0]
        else:
            self.source = source
            self.source_name = ''
        (w, h) = self.source.size
        self.shape = ((h - self.size)//self.stride + 1,
                      (w - self.size)//self.stride + 1)
        self.n_tiles = self.shape[0]*self.shape[1]
            
    def get_tile_rects(self):
        '''
        Yielding iterable of (left, top, right, bottom) tuples for every tile.

        Syntax:
            
            t.get_tile_rects()
        
        '''
        left, top = 0, 0
        while top+self.size <= self.source.size[1]:
            while left+self.size <= self.source.size[0]:
                yield (left, top, left+self.size, top+self.size)
                left += self.stride
            left = 0
            top += self.stride
            
    def get_tile_images(self):
        '''
        Yielding iterable of tile PIL images.

        Syntax:
            
            t.get_tile_images()
        '''
        left, top = 0, 0
        while top+self.size <= self.source.size[1]:
            while left+self.size <= self.source.size[0]:
                yield self.source.crop((left, top, 
                                       left+self.size, top+self.size))
                left += self.stride
            left = 0
            top += self.stride
            
    def write_tiles(self, target_dir='tiles', 
                    rotate=False,
                    flip=False):
        '''
        Writing tiles into separate .png files.
        
        Syntax:
            
            t.write_tiles(target_dir, rotate, flip)
            
        Parameters:
            
            target_dir (str): directory to write tiles in (by default "<rottt>/tiles")
            
            rotate (bool): in addition to original tile, write its copies rotated by 90, 180 and 270 degrees (by default False)
            
            flip (bool):  in addition to original tile, write its copies flipped horizontally and vertically (by default False)

        '''
        if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        tiles = self.get_tile_rects()
        for tile in tiles:
            (left, top, right, bottom) = tile
            tilename = self.source_name+'_x_' +str(left)+'_y_'+str(top)
            crop = self.source.crop(tile)
            crop.save(os.path.join(target_dir, tilename+'.png'))
            if rotate:
                for angle in (90,180,270):
                    copyname = tilename+'_rot_'+str(angle)
                    copy = crop.transpose(rot_angles[angle])
                    copy.save(os.path.join(target_dir, copyname+'.png'))
            if flip:
                for flip_mode in ('vert', 'hor'):
                    copyname = tilename+'_flip_'+flip_mode
                    copy = crop.transpose(flip_modes[flip_mode])
                    copy.save(os.path.join(target_dir, copyname+'.png'))
                if rotate:
                    copy1 = copy.transpose(rot_angles[90])
                    copy1.save(os.path.join(target_dir, copyname+'_rot_90.png'))
                    copy1 = copy.transpose(rot_angles[270])
                    copy1.save(os.path.join(target_dir, copyname+'_rot_270.png'))
        tiles.close()
        
        
    def create_mask(self, tile_indexes):
        tiles = list(self.get_tile_rects())
        mask = Image.new('1', self.source.size, 0)
        draw = ImageDraw.Draw(mask)
        for i in range(len(tiles)):
            if i in tile_indexes:
                draw.rectangle(tiles[i], fill=1)
        del draw
        return mask
                                                
def annotate(root, key_class_dict=None):
    '''
    Tool for annotating images. Requires OpenCV.
    Loops through images in root directory. Press a key to move current image into directory defined by key_class_dict, or "Esc" to break.

    Syntax:
        
        annotate(root, key_class_dict)
        
    Parameters:
        
        root (str): directory with image files.
        
        key_class_dict (dict): dict of format {key_character:directory_name}.
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
    