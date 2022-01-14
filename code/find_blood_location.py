# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:45:04 2020

@author: pang
"""

#import numpy as np
#from PIL import ImageGrab
import cv2
import time
import grabscreen
#import os

def self_blood_count(self_gray):
    self_blood = 0
    count = 0
#    W = self_gray[474].shape
    for self_bd_num in self_gray[476,0:220] :
        
        # self blood gray pixel 80~98
        # 血量灰度值90~99
        if self_bd_num > 90 and self_bd_num < 99  :
            
            self_blood += 1
        count += 1
    print("自己的血量",self_blood)
    return self_blood

def boss_blood_count(boss_gray):
    boss_blood = 0
    count = 0
    for boss_bd_num in boss_gray[2,4:137]:
    # boss blood gray pixel 65~75
    # 血量灰度值65~75 
        # print(boss_bd_num)
        #print("王的血量",boss_bd_num)
        if boss_bd_num > 65 and boss_bd_num < 75 and count<=102:
            
            boss_blood += 1
    count += 1
    print("王的血量",boss_blood)
    return boss_blood

wait_time = 0
L_t = 3

window_size = (320,104,704,448)#(320,104,704,448)
blood_window = (60,86,280,565)#(60,91,280,562)



for i in list(range(wait_time))[::-1]:
    print(i+1)
    time.sleep(1)

last_time = time.time()
while(True):

    printscreen = grabscreen.grab_screen(window_size)
    #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8')\
    #.reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
    #pil格式耗时太长
    
    screen_gray = cv2.cvtColor(grabscreen.grab_screen(blood_window),cv2.COLOR_BGR2GRAY)#灰度图像收集
    # screen_reshape = cv2.resize(screen_gray,(96,86))
    self_blood = self_blood_count(screen_gray)
    boss_blood = boss_blood_count(screen_gray)
    print(screen_gray.shape)
    cv2.line(screen_gray, (4, 2), (137, 2), (255, 255, 255), 2)#boss
    cv2.line(screen_gray, (0, 476), (220, 476), (255, 255, 255), 2)#self
    cv2.imshow('window1',screen_gray)
    #cv2.imshow('window3',printscreen)
    #cv2.imshow('window2',screen_reshape)
    
    #测试时间用
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.waitKey()# 视频结束后，按任意键退出
cv2.destroyAllWindows()
