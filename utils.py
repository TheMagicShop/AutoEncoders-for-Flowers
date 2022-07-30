import os
import pathlib
import random
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

class ImageGenerator():
    def __init__(self, folders, target_size, 
                 batch_size, convert_to_grayscale=False):
        self.folders = folders
        self.target_size = target_size
        self.batch_size = batch_size
        self.convert_to_grayscale = convert_to_grayscale
        self.files = self._build()
        self.start_index = 0
        self.end_index = batch_size
    
    @property
    def _channels(self):
        if self.convert_to_grayscale:
            return 1
        return 3
    
    @property
    def steps_per_epoch(self):
        return len(self.files) // self.batch_size
    
    def _build(self):
        files = []
        for path in self.folders:
            path = pathlib.Path(path)
            fnames = os.listdir(path)
            for fname in fnames:
                files.append(path / fname)
        random.shuffle(files)
        random.shuffle(files)
        
        return files
    
    def reset_index(self):
        random.shuffle(self.files)
        self.start_index = 0
        self.end_index = self.batch_size
        
    def _get_batch(self, fnames):
        X = np.zeros(shape=(self.batch_size, 
                            *self.target_size, 
                            self._channels),
                     dtype='float32')
        for i, fname in enumerate(fnames):
            img = Image.open(fname)
            if self.convert_to_grayscale:
                img = img.convert('L')
            img = img.resize(self.target_size)
            img = np.array(img) / 255.
            if self.convert_to_grayscale:
                img = img[..., np.newaxis]
            if img.ndim != 3:
              continue
            X[i] = img
        
        return X, X
    
    def flow(self, training=True):
        while True:
            fnames = self.files[self.start_index: self.end_index]
            self.start_index += self.batch_size
            self.end_index += self.batch_size
            if self.end_index > len(self.files):
                self.reset_index()
            
            if training:
              yield self._get_batch(fnames)
            else:
              yield self._get_batch(fnames)[0]
        


class TrackImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_sample, show_window=True, 
                 to_file=None, fps=30):
        super(TrackImageCallback, self).__init__()
        self.image_sample = image_sample
        self.size = (256, 256)
        self.at_epoch = 0
        self.show_window = show_window
        self.to_file = to_file
        self.fps = fps
        if to_file is not None:
          self._video_encoding = cv2.VideoWriter_fourcc('M','J','P','G')
          self._video_writer = cv2.VideoWriter(self.to_file, 
                                               self._video_encoding,
                                               fps=fps,
                                               frameSize=self.size)
        
    def on_train_begin(self, logs):
        original_image_upscaled = cv2.resize(self.image_sample, self.size)
        original_image_upscaled = self._to_uint8_bgr(original_image_upscaled)
        if self.show_window:
          cv2.imshow('Tracked Image', original_image_upscaled)
          cv2.waitKey(1)
        if self.to_file is not None:
          for _ in range(2 * self.fps):
            self._video_writer.write(original_image_upscaled)

        
    def on_train_batch_end(self, batch, logs):
        printed = f'epoch: {self.at_epoch}\nbatch: {batch}'
        for key, value in logs.items():
            printed += f'\n{key}: {value:.3}'
        
        predicted = self._predict_and_upscale()
        for i, line in enumerate(printed.split('\n')):
            predicted = cv2.putText(img=predicted,
                                    text=line,
                                    org=(2, 10 + i*20),
                                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=0.55,
                                    thickness=1,
                                    color=(0, 255, 0))
            
        if self.show_window:
          cv2.imshow('Tracked Image', predicted)
          cv2.waitKey(1)
        if self.to_file is not None:
          self._video_writer.write(predicted)
    
    def on_epoch_begin(self, epoch, logs):
        self.at_epoch = epoch
        
    def on_train_end(self, logs):
        if self.show_window:
          cv2.destroyWindow('Tracked Image')
    
    def _predict_and_upscale(self):
        x = np.expand_dims(self.image_sample, axis=0)
        predicted = self.model.predict(x)[0]
        predicted = cv2.resize(predicted, self.size)
        predicted = self._to_uint8_bgr(predicted)
        return predicted
    
    def _to_uint8_bgr(self, image):
      image = image[:, :, ::-1]
      image = (255 * image).astype(np.uint8)
      return image

def get_random_image(species_folder, target_size):
    image_name = random.choice(os.listdir(species_folder))
    image_path = os.path.join(species_folder, image_name)
    image_sample = Image.open(image_path).resize(target_size)
    image_sample = np.array(image_sample) / 255
    return image_sample