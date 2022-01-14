import numpy as np
from grabscreen import grab_screen
import cv2
import time
import directkeys
from getkeys import key_check
from DQN_tensorflow_gpu import DQN
from restart import restart

def pause_game(paused):
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'T' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused

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

def take_action(action):
    if action == 0:     # n_choose
        pass
    elif action == 1:   # j
        directkeys.attack()
    elif action == 2:   # k
        directkeys.jump()
    elif action == 3:   # m
        directkeys.defense()
    elif action == 4:   # r
        directkeys.dodge()


def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, stop, emergence_break):
    # get action reward
    # emergence_break is used to break down training
    # 用於防止出現意外緊急停止訓練防止錯誤訓練數據擾亂神經網路
    if next_self_blood < 1:     # self dead 默認3
        if emergence_break < 2:
            reward = -10#-10
            print('死')
            done = 1
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break
        else:
            reward = -10#-10
            done = 1
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break
    elif next_boss_blood - boss_blood > 40 : #boss dead
    
        if emergence_break < 2:
            reward = 10#20
            print(reward)
            done = 0
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break
        else:
            reward = 20
            done = 0
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break
    else:
        self_blood_reward = 0
        boss_blood_reward = 0
        # print(next_self_blood - self_blood)
        # print(next_boss_blood - boss_blood)
        if next_self_blood - self_blood < -5:#-7
            if stop == 0:
                self_blood_reward = -6#-6
                stop = 1
                # 防止連續取幀時一直計算掉血
        else:
            stop = 0
        if next_boss_blood - boss_blood <= -3:
            boss_blood_reward = 4#4
        # print("self_blood_reward:    ",self_blood_reward)
        # print("boss_blood_reward:    ",boss_blood_reward)
        reward = self_blood_reward + boss_blood_reward
        print(reward)
        done = 0
        emergence_break = 0
        return reward, done, stop, emergence_break
        

DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"
WIDTH = 96#96
HEIGHT = 88#88
window_size = (320,104,704,448)#[(320,104,704,448)384,344  192,172 96,86]
blood_window = (57,86,280,565)#(60,91,280,562)

action_size = 5
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

now_train = 1200#紀錄用
EPISODES = 50 
big_BATCH_SIZE = 64#16
UPDATE_STEP = 50
# times that evaluate the network
num_step = 0
# used to save log graph
target_step = 0
# used to update target Q network
paused = True
# used to stop training

if __name__ == '__main__':
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
    # DQN init
    paused = pause_game(paused)
    # paused at the begin
    emergence_break = 0     
    # emergence_break is used to break down training
    # 用於防止出現意外緊急停止訓練防止錯誤訓練數據擾亂神經網路
    for episode in range(EPISODES):
        screen_gray = cv2.cvtColor(grab_screen(window_size),cv2.COLOR_BGR2GRAY)
        # collect station gray graph
        blood_window_gray = cv2.cvtColor(grab_screen(blood_window),cv2.COLOR_BGR2GRAY)
        # collect blood gray graph for count self and boss blood
        station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
        # change graph to WIDTH * HEIGHT for station input
        blood = grab_screen(blood_window)
        blood = cv2.cvtColor(blood, cv2.COLOR_BGRA2BGR)
        lower_red=np.array([0,43,46])
        upper_red=np.array([10,255,255])
        hsv = cv2.cvtColor(blood, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        blood = cv2.bitwise_and(blood, blood, mask=mask)
        blood_G = cv2.cvtColor(blood, cv2.COLOR_BGR2GRAY)
        # collect blood gray graph for count self and boss blood
        # change graph to WIDTH * HEIGHT for station input
        
        self_blood = self_blood_count(blood_G)
        boss_blood = boss_blood_count(blood_G)
        # count init blood
        target_step = 0
        # used to update target Q network
        done = 0
        total_reward = 0
        stop = 0    
        # 用於防止連續幀重複計算reward
        last_time = time.time()
        while True:
            station = np.array(station).reshape(-1,HEIGHT,WIDTH,1)[0]
            # reshape station for tf input placeholder
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            target_step += 1
            # get the action by state
            action = agent.Choose_Action(station)
            #actionC=['pass','attack','jump','defense','dodge']
            #print(actionC[action])
            take_action(action)
            # take station then the station change
            screen_gray = cv2.cvtColor(grab_screen(window_size),cv2.COLOR_BGR2GRAY)
            blood = grab_screen(blood_window)
            blood = cv2.cvtColor(blood, cv2.COLOR_BGRA2BGR)
            lower_red=np.array([0,43,46])
            upper_red=np.array([10,255,255])
            hsv = cv2.cvtColor(blood, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_red, upper_red)
            blood = cv2.bitwise_and(blood, blood, mask=mask)
            blood_G = cv2.cvtColor(blood, cv2.COLOR_BGR2GRAY)
            # collect blood gray graph for count self and boss blood
            next_station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
            next_station = np.array(next_station).reshape(-1,HEIGHT,WIDTH,1)[0]
            next_boss_blood = boss_blood_count(blood_G)
            next_self_blood = self_blood_count(blood_G)
            reward, done, stop, emergence_break = action_judge(boss_blood, next_boss_blood,
                                                               self_blood, next_self_blood,
                                                               stop, emergence_break)
            # get action reward
            if emergence_break == 100:
                # emergence break , save model and paused
                # 遇到緊急情況，保存資料，並且暫停
                print("emergence_break")
                agent.save_model()
                paused = True
            agent.Store_Data(station, action, reward, next_station, done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                # save loss graph
                # print('train')
                agent.Train_Network(big_BATCH_SIZE, num_step)
            if target_step % UPDATE_STEP == 0:
                agent.Update_Target_Network()
                # update target Q network
            station = next_station
            self_blood = next_self_blood
            boss_blood = next_boss_blood
            total_reward += reward
            paused = pause_game(paused)
            if done == 1:
                break
        if episode % 5 == 0:
            agent.save_model()
            # save model
        print('episode: ', episode, 'Evaluation Average Reward:', total_reward/target_step)
        restart()

        
            
            
            
            
            
        
        
    
    
