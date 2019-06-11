
# coding: utf-8

# In[1]:


import cv2
import ast
import os
import json
import numpy as np
import tensorflow as tf
import math
import collections
import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread
import Network as net


# In[2]:


def make_batch(img_path, anno_data, batch_size = 16):
    num_of_data = len(img_path)
    index = np.arange(0, num_of_data)
    np.random.shuffle(index)
    index = index[:batch_size]
    
    shuffled_img_data = [img_path[i] for i in index]
    #shuffled_anno_data = [anno_data[j] for j in index]
    shuffled_anno_data = [[anno_data[j:j+1]][0][0] for j in index]
    #shuffled_anno_data = [anno_data[j:j+1][0][0][0] for j in index]
    
    return np.asarray(shuffled_img_data), np.asarray(shuffled_anno_data)

def load_data():
    joint_result = np.load('./dataset/Hands/hand_labels/Annotation/hands_joint_data.npy')
    train_file_list = np.load('./dataset/Hands/hand_labels/Annotation/Resized_hands_img_data.npy')
    return train_file_list, joint_result

def path_to_image(img_path, batch_size):
    #buffer 선언
    image_data = np.zeros((batch_size, 356, 356, 3), np.uint8)
    
    index = 0
    for img in (img_path):
        image_data[index] = cv2.imread(img)
        index = index + 1

    return image_data


# In[5]:


class OpenPose_Demo_Hand():
    def __init__(self, sess, batch_size):
        self.sess = sess
        self.batch_size = batch_size
        self.img_input = tf.placeholder(dtype=tf.float32, shape=[None, 356, 356, 3])
        self.confidencemap = tf.placeholder(dtype = tf.float32, shape = [None, 356, 356, 22])
        self.PAFs = tf.placeholder(dtype = tf.float32, shape = [None, 356, 356, 44])
        
        self.model()
        print("open pose demo init complete")
        
    def model(self):
        """self.Feature = net.block_vgg_19_hand(self.img_input)#stage0_data - None, 44, 44, 512
        
        self.stage1_branch = net.block_stage_1_hand(self.Feature)#stage1_branch - None, 44, 44, 34
        self.stage1_data = tf.concat([self.stage1_branch, self.Feature], 3)
        
        self.stage2_branch = net.block_stage_2_hand(self.stage1_data)#stage2_branch - None, 44, 44, 34
        self.stage2_data = tf.concat([self.stage2_branch, self.Feature], 3)
        
        self.stage3_branch = net.block_stage_3_hand(self.stage2_data)#stage3_branch - None, 44, 44, 34
        self.stage3_data = tf.concat([self.stage3_branch, self.Feature], 3)
        
        self.stage4_branch = net.block_stage_4_hand(self.stage3_data)#stage4_branch - None, 44, 44, 34
        self.stage4_data = tf.concat([self.stage4_branch, self.Feature], 3)
        
        self.stage5_branch = net.block_stage_5_hand(self.stage4_data)#stage5_branch - None, 44, 44, 17
        #self.stage5_data = tf.concat([self.stage5_branch, stage0_data], 3)
        
        self.stage6_branch = net.block_stage_6_hand(self.stage5_branch)#stage6_branch - None, 44, 44, 17
        #self.stage6_data = tf.concat([self.stage6_branch, stage0_data], 3)"""
        self.Feature = net.block_vgg_19_hand(self.img_input)#stage0_data - None, 44, 44, 512
        #self.Feature = net.block_vgg_19_hand(self.img_input)#stage0_data - None, 44, 44, 512
        
        self.stage1_branch = net.block_stage_1_hand(self.Feature)#stage1_branch - None, 44, 44, 34
        #self.stage1_branch = net.block_stage_1(self.Feature)#stage1_branch - None, 44, 44, 34
        self.stage1_data = tf.concat([self.stage1_branch, self.Feature], 3)
        
        self.stage2_branch = net.block_stage_2_hand(self.stage1_data)#stage2_branch - None, 44, 44, 34
        #self.stage2_branch = net.block_stage_2(self.stage1_data)#stage2_branch - None, 44, 44, 34
        self.stage2_data = tf.concat([self.stage2_branch, self.Feature], 3)
        
        self.stage3_branch = net.block_stage_3_hand(self.stage2_data)#stage3_branch - None, 44, 44, 34
        #self.stage3_branch = net.block_stage_3(self.stage2_data)#stage3_branch - None, 44, 44, 34
        self.stage3_data = tf.concat([self.stage3_branch, self.Feature], 3)
        
        self.stage4_branch = net.block_stage_4_hand(self.stage3_data)#stage4_branch - None, 44, 44, 34
        #self.stage4_branch = net.block_stage_4(self.stage3_data)#stage4_branch - None, 44, 44, 34
        self.stage4_data = tf.concat([self.stage4_branch, self.Feature], 3)
        
        self.stage5_branch = net.block_stage_5_hand(self.stage4_data)#stage5_branch - None, 44, 44, 17
        #self.stage5_branch = net.block_stage_5(self.stage4_data)#stage5_branch - None, 44, 44, 17
        #self.stage5_data = tf.concat([self.stage5_branch, stage0_data], 3)
        
        self.stage6_branch = net.block_stage_6_hand(self.stage5_branch)#stage6_branch - None, 44, 44, 17
        #self.stage6_branch = net.block_stage_6(self.stage5_branch)#stage6_branch - None, 44, 44, 17
        #self.stage6_data = tf.concat([self.stage6_branch, stage0_data], 3)
        
    
    def demo_test(self, input_data):
        input_img = input_data
        SAVE_PATH = "C:/Users/SimJaekyu/Documents/Jupyter Notebook/Pose_Estimation_with_Hand/Weight_Hand/Weight.ckpt"
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        try:
            saver.restore(self.sess, SAVE_PATH)
            print("Training Weight load")
        except:
            print("No Training Weight exit")
            
        confidencemap, PAFs = self.sess.run([self.stage6_branch, self.stage4_branch], feed_dict = {self.img_input : input_img})
        
        return confidencemap, PAFs
            

