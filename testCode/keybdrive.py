import  RPi.GPIO as GPIO
import time
import pygame
from pygame.locals import *
import sys

pygame.init()
screen=pygame.display.set_mode((75,75))  #windows size

PWMA = 18
AIN1 = 22
AIN2 = 27

PWMB = 23
BIN1 = 25
BIN2 = 24

def t_up(speedL,speedR):
        L_Motor.ChangeDutyCycle(speedL)
        GPIO.output(AIN2,False)#AIN2
        GPIO.output(AIN1,True) #AIN1

        R_Motor.ChangeDutyCycle(speedR)
        GPIO.output(BIN2,False)#BIN2
        GPIO.output(BIN1,True) #BIN1

def t_stop(t_time):
        L_Motor.ChangeDutyCycle(0)
        GPIO.output(AIN2,False)#AIN2
        GPIO.output(AIN1,False) #AIN1

        R_Motor.ChangeDutyCycle(0)
        GPIO.output(BIN2,False)#BIN2
        GPIO.output(BIN1,False) #BIN1
        time.sleep(t_time)

def t_down(speedL,speedR):
        L_Motor.ChangeDutyCycle(speedL)
        GPIO.output(AIN2,True)#AIN2
        GPIO.output(AIN1,False) #AIN1

        R_Motor.ChangeDutyCycle(speedR)
        GPIO.output(BIN2,True)#BIN2
        GPIO.output(BIN1,False) #BIN1

def t_left(speedL,speedR):
        L_Motor.ChangeDutyCycle(speedL)
        GPIO.output(AIN2,True)#AIN2
        GPIO.output(AIN1,False) #AIN1

        R_Motor.ChangeDutyCycle(speedR)
        GPIO.output(BIN2,False)#BIN2
        GPIO.output(BIN1,True) #BIN1

def t_right(speedL,speedR):
        L_Motor.ChangeDutyCycle(speedL)
        GPIO.output(AIN2,False)#AIN2
        GPIO.output(AIN1,True) #AIN1

        R_Motor.ChangeDutyCycle(speedR)
        GPIO.output(BIN2,True)#BIN2
        GPIO.output(BIN1,False) #BIN1

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN2,GPIO.OUT)
GPIO.setup(AIN1,GPIO.OUT)
GPIO.setup(PWMA,GPIO.OUT)
GPIO.setup(BIN1,GPIO.OUT)
GPIO.setup(BIN2,GPIO.OUT)
GPIO.setup(PWMB,GPIO.OUT)
L_Motor=GPIO.PWM(PWMA,100)
L_Motor.start(0)
R_Motor=GPIO.PWM(PWMB,100)
R_Motor.start(0)
try:
    while True:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_w]:
                    t_up(100,100)
                    print("forward")
                elif keys[pygame.K_s]:
                    t_down(100,100)
                    print("back")
                elif keys[pygame.K_q]:
                    t_up(50,100)
                    print("left")
                elif keys[pygame.K_e]:
                    t_up(100,50)
                    print("right")
                elif keys[pygame.K_z]:
                    t_down(50,100)
                    print("left back")
                elif keys[pygame.K_c]:
                    t_down(100,50)
                    print("right back")
                elif keys[pygame.K_a]:
                    t_left(100,100)
                    print("turn left")
                elif keys[pygame.K_d]:
                    t_right(100,100)
                    print("turn right")
                    break
            elif event.type == KEYUP:
                t_stop(0)
                print("stop")
except KeyboardInterrupt:
    GPIO.cleanup()         
