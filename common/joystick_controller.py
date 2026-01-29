import threading
import time
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
from loguru import logger

class JoyStickController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.connect_state = False
        self._lock = threading.Lock()
        self.stick_val = (0, 0, 0, 0)
        self.button_val = {
                        "A": 0,
                        "B": 0,
                        "X": 0,
                        "Y": 0,
                        "LB":0,
                        "RB":0,
                        "up":0,
                        "down":0,
                        "right":0,
                        "left":0,
                        }
        if pygame.joystick.get_count() == 0:
            logger.warning(f"joystick on pc not detected")
            # import pdb;pdb.set_trace()
            return
        logger.info(f"joystick on pc detected")

        self.connect_state = True
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop)
        self._thread.start()

    def _poll_loop(self):
        while self._running:
            pygame.event.pump()
            lx = self.joystick.get_axis(0) if abs(self.joystick.get_axis(0)) > 0.01 else 0
            ly = self.joystick.get_axis(1) if abs(self.joystick.get_axis(1)) > 0.01 else 0
            rx = self.joystick.get_axis(2) if abs(self.joystick.get_axis(2)) > 0.01 else 0
            ry = self.joystick.get_axis(3) if abs(self.joystick.get_axis(3)) > 0.01 else 0
            
            lb, rb = self.joystick.get_button(9), self.joystick.get_button(10)
            self.button_val["LB"] = lb
            self.button_val["RB"] = rb
            
            self.button_val["A"] = self.joystick.get_button(0)
            self.button_val["B"] = self.joystick.get_button(1)
            self.button_val["X"] = self.joystick.get_button(2)
            self.button_val["Y"] = self.joystick.get_button(3)
            
            self.button_val["up"] = self.joystick.get_button(11)
            self.button_val["down"] = self.joystick.get_button(12)
            self.button_val["left"] = self.joystick.get_button(13)
            self.button_val["right"] = self.joystick.get_button(14)
            
            with self._lock:
                self.stick_val = (lx, ly, rx, ry)
            time.sleep(0.02)  # 防止 CPU 过载

    def get_stick_val(self):
        with self._lock:
            return self.stick_val

    def get_button_val(self):
        with self._lock:
            return self.button_val
        
    def is_connected(self):
        return self.connect_state
    
    def stop(self):
        self._running = False
        self._thread.join()
        pygame.quit()
