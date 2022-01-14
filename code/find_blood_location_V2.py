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

def self_blood_count(self_gray):
    self_blood = 0
#    W = self_gray[474].shape
    for self_bd_num in self_gray[476,2:220] :
        if self_bd_num >=110:
            break
        elif self_bd_num > 62 and self_bd_num < 90 :
            self_blood += 1
        #print(self_gray[476,1:220])
    print("自己的血量",self_blood)
    return self_blood

def boss_blood_count(boss_gray):
    boss_blood = 0        
    for boss_bd_num in boss_gray[4,5:143]:
        if boss_bd_num >=85 and boss_bd_num <=110:
            break
        elif np.max(boss_gray[4,5:143]) < 70:
            boss_blood = 0
        else:
            boss_blood += 1
    print(boss_gray[4,5:143])
    print("王的血量",boss_blood)
    return boss_blood

wait_time = 0
L_t = 3

window_size = (320,104,704,448)#(320,104,704,448)
blood_window = (57,86,280,565)#(60,91,280,562) (60,86,280,565)



for i in list(range(wait_time))[::-1]:
    print(i+1)
    time.sleep(1)

last_time = time.time()
while(True):

    printscreen = grabscreen.grab_screen(window_size)
    screen_gray = cv2.cvtColor(grabscreen.grab_screen(blood_window),cv2.COLOR_BGR2GRAY)
    blood = grabscreen.grab_screen(blood_window)
    blood = cv2.cvtColor(blood, cv2.COLOR_BGRA2BGR)
    b, g ,r =cv2.split(blood)
    self_blood = self_blood_count(b)
    boss_blood = boss_blood_count(b) 
    
    #cv2.line(screen_gray, (4, 2), (140, 2), (255, 255, 255), 1)#boss
    #cv2.line(b, (1, 476), (220, 476), (255, 255, 255), 2)#self
    cv2.imshow('window1',b)
    
    #测试时间用
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.waitKey()# 视频结束后，按任意键退出
cv2.destroyAllWindows()
