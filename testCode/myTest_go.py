#-*- coding:UTF-8 -*-
import RPi.GPIO as GPIO
import time
import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from email.mime.image import MIMEImage
from time import sleep
from pip._vendor import requests
import sys
import pygame
from pygame.locals import *

pygame.init()
screen=pygame.display.set_mode((75,75))  #windows size

# 小车电机引脚定义
IN1 = 20 # 左前轮
IN2 = 21 # 左后轮
IN3 = 19 # 右前轮
IN4 = 26 # 右后轮
ENA = 16 # 左电机控速PWMA 连接Raspberry
ENB = 13 # 右电机控速PWMB 连接Raspberry


# 电机引脚初始化操作
def motor_init():
    global pwm_ENA
    global pwm_ENB
    GPIO.setup(ENA,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(IN1,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN2,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(ENB,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(IN3,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN4,GPIO.OUT,initial=GPIO.LOW)
    # 设置pwm引脚和频率为2000hz
    pwm_ENA = GPIO.PWM(ENA, 2000)
    pwm_ENB = GPIO.PWM(ENB, 2000)
    # 启动PWM设置占空比为100（0--100）
    pwm_ENA.start(0)
    pwm_ENB.start(0)

#小车前进
def t_up(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)


#小车后退
def t_down(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)


#小车左转
def t_left(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)


#小车右转
def t_right(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)

"""
#小车原地左转
def spin_left(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)


#小车原地右转
def spin_right(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)
"""

#小车停止
def t_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(0)
    pwm_ENB.ChangeDutyCycle(0)

try:
    motor_init()
    while True:
        for event in pygame.event.get():            
            if event.type == KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_w]:
                    t_up(100,100)
                    time.sleep(0.05)
                    print("forward")
                elif keys[pygame.K_s]:
                    t_down(100,100)
                    time.sleep(0.05)
                    print("back")
                elif keys[pygame.K_q]:
                    t_up(50,100)
                    time.sleep(0.05)
                    print("left")
                elif keys[pygame.K_e]:
                    t_up(100,50)
                    time.sleep(0.05)
                    print("right")
                elif keys[pygame.K_z]:
                    t_down(50,100)
                    time.sleep(0.05)
                    print("left back")
                elif keys[pygame.K_c]:
                    t_down(100,50)
                    time.sleep(0.05)
                    print("right back")
                elif keys[pygame.K_a]:
                    t_left(100,100)
                    time.sleep(0.05)
                    print("turn left")
                elif keys[pygame.K_d]:
                    t_right(100,100)
                    time.sleep(0.05)
                    print("turn right")
                    break
            elif event.type == KEYUP:
                t_stop()    
                time.sleep(0.05)           
                print("stop")
except KeyboardInterrupt:
    # GPIO.cleanup()
    pass
pwm_ENA.STOP()
pwm_ENB.STOP()
GPIO.clearnup()
