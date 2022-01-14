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
import numpy as np
#import os

def boss_dogde_count(boss_dogde_gray):
    boss_dogde = 0
    if np.argmax(boss_dogde_gray[0]) <=10:
        boss_dogde = 0
    else:
        boss_dogde = np.argmax(boss_dogde_gray[476,4:193]) 
    #print(boss_dogde_gray[476,4:193])
    print("王的架勢條",boss_dogde)
    return boss_dogde
wait_time = 0
L_t = 3

dogde_window = (513,74,630,80)#(60,91,280,562)(最左的座標,最上的座標,最右的座標,最下的座標)

for i in list(range(wait_time))[::-1]:
    print(i+1)
    time.sleep(1)

last_time = time.time()
while(True):
    dogde = grabscreen.grab_screen(dogde_window)
    dogde = cv2.cvtColor(dogde, cv2.COLOR_BGRA2BGR)
    lower_yellow = np.array([11,43,46])
    upper_yellow = np.array([25,255,255])
    hsv_dogde = cv2.cvtColor(dogde, cv2.COLOR_BGR2HSV)
    mask_dogde = cv2.inRange(hsv_dogde, lower_yellow, upper_yellow)
    dogde = cv2.bitwise_and(dogde, dogde, mask=mask_dogde)
    dogde_G = cv2.cvtColor(dogde, cv2.COLOR_BGR2GRAY)
    print(dogde_G[0])
    #boss_dogde = boss_dogde_count(dogde_G) 

    cv2.imshow('window1',dogde_G)
    
    #测试时间用
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.waitKey()# 视频结束后，按任意键退出
cv2.destroyAllWindows()
