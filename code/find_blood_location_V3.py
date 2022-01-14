# -*- coding: utf-8 -*-

import cv2
import time
import grabscreen
import numpy as np

def self_blood_count(self_gray):
    self_blood = 0
    if self_gray[476,4] <=1:
        self_blood = 0
    else:
        self_blood = np.argmax(self_gray[476,4:193]) 
    #print(self_gray[476,4:193])
    print("自己的血量",self_blood)
    return self_blood

def boss_blood_count(boss_gray):
    boss_blood = 0
    if boss_gray[4,6] <=1:
        boss_blood = 0
    else:
        boss_blood = np.argmax(boss_gray[4,6:143]) 
    #print(boss_gray[4,6:143])
    print("王的血量",boss_blood)
    return boss_blood

wait_time = 0
L_t = 3

window_size = (320,104,704,448)#(320,104,704,448)(最左的座標,最上的座標,最右的座標,最下的座標)
blood_window = (57,86,280,565)#(60,91,280,562) (60,86,280,565)
all_window = (57,86,1010,615)



for i in list(range(wait_time))[::-1]:
    print(i+1)
    time.sleep(1)

last_time = time.time()
while(True):
    window = grabscreen.grab_screen(all_window)

    printscreen = grabscreen.grab_screen(window_size)
    screen_gray = cv2.cvtColor(grabscreen.grab_screen(blood_window),cv2.COLOR_BGR2GRAY)
    blood = grabscreen.grab_screen(blood_window)
    blood = cv2.cvtColor(blood, cv2.COLOR_BGRA2BGR)
    lower_red=np.array([0,43,46])
    upper_red=np.array([10,255,255])
    hsv = cv2.cvtColor(blood, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    blood = cv2.bitwise_and(blood, blood, mask=mask)
    blood_G = cv2.cvtColor(blood, cv2.COLOR_BGR2GRAY)
    boss_blood = boss_blood_count(blood_G) 
    self_blood = self_blood_count(blood_G)

    red_color = (0, 0, 255)
    cv2.line(window, (4, 2), (140, 2), (255, 0, 0), 2)#boss
    cv2.line(window, (4, 476), (193, 476), (0, 255, 0), 2)#self
    cv2.rectangle(window, (320, 104), (704, 448), red_color, 2, cv2.LINE_AA)
    cv2.imshow('window1',window)
    
    #测试时间用
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.waitKey()# 视频结束后，按任意键退出
cv2.destroyAllWindows()
