#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:46:52 2017
@author: mpekmezci
"""
import math
import tensorflow as tf
#import urllib3
import tarfile
import csv
import glob
import sys
import os
import argparse
import sys
import tempfile
import numpy as np
import librosa
import pandas as pd
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13

script_dir=os.path.dirname(os.path.realpath(__file__))
main_data_dir = script_dir+'/../data/'
raw_data_dir = main_data_dir+'/0.raw/UrbanSound8K/audio'
csv_data_dir=main_data_dir+"/1.csv"
#fold_dirs = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
fold_dirs = ['fold1']
fold_data_dictionary=dict()



# 1 RECORD is 4 seconds = 4 x sampling rate double values = 4 x 22050 = 88200 = (2^3) x ( 3^2) x (5^2) x (7^2)
SOUND_RECORD_SAMPLING_RATE=22050



def augment_data():
    global csv_data_dir,fold_dirs
    for fold in fold_dirs:
      print("Augmenting Data in the Fold :"+fold)
      csvData=np.array(np.loadtxt(open(csv_data_dir+"/"+fold+".csv", "rb"), delimiter=","))
      #for i in range(csvData.shape[0]) :
      for i in range(20) :
          csvDataLine=csvData[i]

          csvDataLineX=csvDataLine[:4*SOUND_RECORD_SAMPLING_RATE]
          librosa.output.write_wav('/tmp/'+str(i)+'.base.wav', csvDataLineX , SOUND_RECORD_SAMPLING_RATE)

          TRANSLATION_FACTOR=2000
          augmentDataX=augment_translate(csvDataLineX,TRANSLATION_FACTOR)
          librosa.output.write_wav('/tmp/'+str(i)+'.translation.wav',  augmentDataX , SOUND_RECORD_SAMPLING_RATE)


          SPEED_FACTOR=1.1
          augmentDataX=augment_speedx(csvDataLineX,SPEED_FACTOR)
          librosa.output.write_wav('/tmp/'+str(i)+'.speedx.wav',  augmentDataX , SOUND_RECORD_SAMPLING_RATE)

          PITCH_SHIFT_FACTOR=1.1
          augmentDataX=augment_pitchshift(csvDataLineX,PITCH_SHIFT_FACTOR)
          librosa.output.write_wav('/tmp/'+str(i)+'.pitchshift.wav',  augmentDataX , SOUND_RECORD_SAMPLING_RATE)

def augment_speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]


def augment_stretch(sound_array, f, window_size, h):
    """ Stretches the sound by a factor `f` """

    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros( int(len(sound_array) /f + window_size))

    for i in np.arange(0, len(sound_array)-(window_size+h), h*f):

        # two potentially overlapping subarrays

        i=int(i)
        a1 = sound_array[i: i + window_size]
        a2 = sound_array[i + h: i + window_size + h]

        # resynchronize the second array on the first
        s1 =  np.fft.fft(hanning_window * a1)
        s2 =  np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % 2*np.pi
        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))

        # add to result
        i2 = int(i/f)
        result[i2 : i2 + window_size] = result[i2 : i2 + window_size] +  hanning_window*a2_rephased

    result = ((2**(16-4)) * result/result.max()) # normalize (16bit)

    return result.astype('int16')


def augment_pitchshift(snd_array, n, window_size=2**13, h=2**11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2**(1.0 * n / 12.0)
    stretched = augment_stretch(snd_array, 1.0/factor, window_size, h)
    return augment_speedx(stretched[window_size:], factor)


def augment_translate(snd_array, n):
    """ Translates the sound wave by n indices, fill the first n elements of the array with random numbers """
    new_array=np.zeros(len(snd_array))
    for i in range(snd_array.shape[0]):
       if i+n == len(new_array) :
          break
       new_array[i+n]=snd_array[i] 
    return new_array




augment_data()
