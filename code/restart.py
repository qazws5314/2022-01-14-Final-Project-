# -*- coding: utf-8 -*-

import directkeys
import time

def restart():
    print("死,restart")
    time.sleep(8)
    directkeys.lock_vision()
    time.sleep(0.2)
    directkeys.attack()
    print("開始新一輪")
  
if __name__ == "__main__":  
    restart()

