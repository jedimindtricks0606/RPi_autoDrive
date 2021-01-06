# -*- coding:UTF-8 -*-
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
import numpy as np
import cv2
import os

class CollectData(object):

    def __init__(self):
        #GPIO.cleanup()
        pygame.init()
        #screen=pygame.display.set_mode((75,75))  #windows size

        # 小车电机引脚定义
        self.IN1 = 20  # 左前轮
        self.IN2 = 21  # 左后轮
        self.IN3 = 19  # 右前轮
        self.IN4 = 26  # 右后轮
        self.ENA = 16  # 左电机控速PWMA 连接Raspberry
        self.ENB = 13  # 右电机控速PWMB 连接Raspberry
        
        # 设置GPIO 口为BCM 编码方式
        GPIO.setmode(GPIO.BCM)
        # 忽略警告信息
        GPIO.setwarnings(False)
        
        GPIO.setup(self.ENA,GPIO.OUT,initial=GPIO.HIGH)
        GPIO.setup(self.IN1,GPIO.OUT,initial=GPIO.LOW)
        GPIO.setup(self.IN2,GPIO.OUT,initial=GPIO.LOW)
        GPIO.setup(self.ENB,GPIO.OUT,initial=GPIO.HIGH)
        GPIO.setup(self.IN3,GPIO.OUT,initial=GPIO.LOW)
        GPIO.setup(self.IN4,GPIO.OUT,initial=GPIO.LOW)
        # 设置pwm引脚和频率为2000hz
        self.pwm_ENA = GPIO.PWM(self.ENA, 2000)
        self.pwm_ENB = GPIO.PWM(self.ENB, 2000)
        # 启动PWM设置占空比为100（0--100）
        self.pwm_ENA.start(0)
        self.pwm_ENB.start(0)

        self.save_img = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 320)
        self.cap.set(4, 240)

        # create labels
        self.k = np.zeros((3, 3), 'float')
        for i in range(3):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 3), 'float')

        screen = pygame.display.set_mode((320, 240))

        self.collect_image()

    # 电机引脚初始化操作
    def motor_init(self):
        # global pwm_ENA
        # global pwm_ENB
        GPIO.setup(ENA, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(ENB, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)
        # 设置pwm引脚和频率为2000hz
        self.pwm_ENA = GPIO.PWM(ENA, 2000)
        self.pwm_ENB = GPIO.PWM(ENB, 2000)
        # 启动PWM设置占空比为100（0--100）
        pwm_ENA.start(0)
        pwm_ENB.start(0)

    # 小车前进
    def t_up(self, leftspeed, rightspeed):
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        pwm_ENA.ChangeDutyCycle(leftspeed)
        pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车后退
    def t_down(self, leftspeed, rightspeed):
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
        pwm_ENA.ChangeDutyCycle(leftspeed)
        pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车左转
    def t_left(self, leftspeed, rightspeed):
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        pwm_ENA.ChangeDutyCycle(leftspeed)
        pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车右转
    def t_right(self, leftspeed, rightspeed):
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

    # 小车停止
    def t_stop(self):
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)
        pwm_ENA.ChangeDutyCycle(0)
        pwm_ENB.ChangeDutyCycle(0)

    def collect_image(self):
        frame = 1
        saved_frame = 0
        total_frame = 0
        label_0 = 0
        label_1 = 1
        label_2 = 2
        # collect images for training
        print('Start collecting images...')

        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 3), 'float')

        # collect cam frames
        try:
            while self.save_img:
                ret, cam = self.cap.read()
                roi = cam[120:240, :]
                # image = cv2.imdecode(np.fromstring(cam,dtype=np.uint8),cv2.CV_LOAD_IMAGE_GRAYSCALE)
                gauss = cv2.GaussianBlur(roi, (5, 5), 0)
                gray = cv2.cvtColor(cam, cv2.COLOR_RGB2GRAY)
                # dst = cv2.Canny(gray,50,50)
                # ret,thresh1 = cv2.threshold(gray,90,255,cv2.THRESH_BINARY)
                ret, th3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # cv2.imwrite('./training_images/frame{:>05}.jpg'.format(frame),dst)

                # cv2.imshow('cam',cam)
                # cv2.imshow('dst',dst)
                # cv2.imshow('gray',gray)
                # cv2.imshow('roi',roi)
                # cv2.imshow('thresh1',thresh1)
                # cv2.imshow('th3',th3)
                temp_array = roi.reshape(1, 38400).astype(np.float32)
                frame += 1
                total_frame += 1

                if cv2.waitKey(1) & 0xFF == ord('l'):
                    print("exit")
                    break
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_w]:
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[0]))
                            saved_frame += 1
                            self.t_up(100, 100)
                            cv2.imwrite('./training_images/{:1}_frame{:>05}.jpg'.format(label_0, saved_frame), th3)
                            print("forward")
                            print(saved_frame)

                        elif keys[pygame.K_q]:
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[1]))
                            saved_frame += 1
                            self.t_up(50, 100)
                            cv2.imwrite('./training_images/{:1}_frame{:>05}.jpg'.format(label_1, saved_frame), th3)
                            print("left")
                            print(saved_frame)

                        elif keys[pygame.K_e]:
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[2]))
                            saved_frame += 1
                            self.t_up(100, 50)
                            cv2.imwrite('./training_images/{:1}_frame{:>05}.jpg'.format(label_2, saved_frame), th3)
                            print("right")
                            print(saved_frame)

                        elif keys[pygame.K_UP]:
                            self.t_up(100, 100)
                            print("forward")

                        elif keys[pygame.K_s]:
                            self.t_down(100, 100)
                            print("back")
                        elif keys[pygame.K_DOWN]:
                            self.t_down(100, 100)
                            print("back")

                        elif keys[pygame.K_LEFT]:
                            self.t_up(50, 100)
                            print("left")

                        elif keys[pygame.K_RIGHT]:
                            self.t_up(100, 50)
                            print("right")

                        elif keys[pygame.K_a]:
                            self.t_left(100, 100)
                            print("turn left")

                        elif keys[pygame.K_d]:
                            self.t_right(100, 100)
                            print("turn right")

                        elif keys[pygame.K_z]:
                            self.t_down(50, 100)
                            print("left back")

                        elif keys[pygame.K_c]:
                            self.t_down(100, 50)
                            print("right back")

                        elif keys[pygame.K_p]:
                            print('exit')
                            self.save_img = False
                            break

                    elif event.type == KEYUP:
                        self.t_stop(0)
                        print("stop")

            # save training images and labels

            train = image_array[1:, :]
            train_labels = label_array[1:, :]
            # save training data as a numpy file

            file_name = str(int(time.time()))
            directory = "training_data"
            if not os.path.exists(directory):
                os.makedirs(directory)
            try:
                np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
            except IOError as e:
                print(e)
            e2 = cv2.getTickCount()

            time0 = (e2 - e1) / cv2.getTickFrequency()
            print('Video duration:', time0)
            print('Streaming duration:', time0)

            print((train.shape))
            print((train_labels.shape))
            print('Total frame:', total_frame)
            print('Saved frame:', saved_frame)
            print('Dropped frame', total_frame - saved_frame)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    CollectData()

# import numpy as np
# import cv2
# import pygame
# from pygame.locals import *
# import time
# import os
# import RPi.GPIO as GPIO
#
#
# class CollectData(object):
#
#     def __init__(self):
#         # set IO and GPIO mod
#         GPIO.cleanup()
#         # pygame init
#         pygame.init()
#
#         self.PWMA = 18
#         self.AIN1 = 22
#         self.AIN2 = 27
#
#         self.PWMB = 23
#         self.BIN1 = 25
#         self.BIN2 = 24
#
#         GPIO.setwarnings(False)
#         GPIO.setmode(GPIO.BCM)
#         GPIO.setup(self.AIN2, GPIO.OUT)
#         GPIO.setup(self.AIN1, GPIO.OUT)
#         GPIO.setup(self.PWMA, GPIO.OUT)
#         GPIO.setup(self.BIN1, GPIO.OUT)
#         GPIO.setup(self.BIN2, GPIO.OUT)
#         GPIO.setup(self.PWMB, GPIO.OUT)
#         self.L_Motor = GPIO.PWM(self.PWMA, 100)
#         self.L_Motor.start(0)
#         self.R_Motor = GPIO.PWM(self.PWMB, 100)
#         self.R_Motor.start(0)
#
#         self.save_img = True
#         self.cap = cv2.VideoCapture(0)
#         self.cap.set(3, 320)
#         self.cap.set(4, 240)
#
#         # create labels
#         self.k = np.zeros((3, 3), 'float')
#         for i in range(3):
#             self.k[i, i] = 1
#         self.temp_label = np.zeros((1, 3), 'float')
#
#         screen = pygame.display.set_mode((320, 240))
#
#         self.collect_image()
#
#     def t_up(self, speedL, speedR):
#         self.L_Motor.ChangeDutyCycle(speedL)
#         GPIO.output(self.AIN2, False)  # AIN2
#         GPIO.output(self.AIN1, True)  # AIN1
#
#         self.R_Motor.ChangeDutyCycle(speedR)
#         GPIO.output(self.BIN2, False)  # BIN2
#         GPIO.output(self.BIN1, True)  # BIN1
#
#     def t_stop(self, t_time):
#         self.L_Motor.ChangeDutyCycle(0)
#         GPIO.output(self.AIN2, False)  # AIN2
#         GPIO.output(self.AIN1, False)  # AIN1
#
#         self.R_Motor.ChangeDutyCycle(0)
#         GPIO.output(self.BIN2, False)  # BIN2
#         GPIO.output(self.BIN1, False)  # BIN1
#         time.sleep(t_time)
#
#     def t_down(self, speedL, speedR):
#         self.L_Motor.ChangeDutyCycle(speedL)
#         GPIO.output(self.AIN2, True)  # AIN2
#         GPIO.output(self.AIN1, False)  # AIN1
#
#         self.R_Motor.ChangeDutyCycle(speedR)
#         GPIO.output(self.BIN2, True)  # BIN2
#         GPIO.output(self.BIN1, False)  # BIN1
#
#     def t_left(self, speedL, speedR):
#         self.L_Motor.ChangeDutyCycle(speedL)
#         GPIO.output(self.AIN2, True)  # AIN2
#         GPIO.output(self.AIN1, False)  # AIN1
#
#         self.R_Motor.ChangeDutyCycle(speedR)
#         GPIO.output(self.BIN2, False)  # BIN2
#         GPIO.output(self.BIN1, True)  # BIN1
#
#     def t_right(self, speedL, speedR):
#         self.L_Motor.ChangeDutyCycle(speedL)
#         GPIO.output(self.AIN2, False)  # AIN2
#         GPIO.output(self.AIN1, True)  # AIN1
#
#         self.R_Motor.ChangeDutyCycle(speedR)
#         GPIO.output(self.BIN2, True)  # BIN2
#         GPIO.output(self.BIN1, False)  # BIN1
#
#     def collect_image(self):
#         frame = 1
#         saved_frame = 0
#         total_frame = 0
#         label_0 = 0
#         label_1 = 1
#         label_2 = 2
#         # collect images for training
#         print('Start collecting images...')
#
#         e1 = cv2.getTickCount()
#         image_array = np.zeros((1, 38400))
#         label_array = np.zeros((1, 3), 'float')
#
#         # collect cam frames
#         try:
#
#             while self.save_img:
#                 ret, cam = self.cap.read()
#                 roi = th3[120:240, :]
#                 # image = cv2.imdecode(np.fromstring(cam,dtype=np.uint8),cv2.CV_LOAD_IMAGE_GRAYSCALE)
#                 gauss = cv2.GaussianBlur(roi, (5, 5), 0)
#                 gray = cv2.cvtColor(cam, cv2.COLOR_RGB2GRAY)
#                 # dst = cv2.Canny(gray,50,50)
#                 # ret,thresh1 = cv2.threshold(gray,90,255,cv2.THRESH_BINARY)
#                 ret, th3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#                 # cv2.imwrite('./training_images/frame{:>05}.jpg'.format(frame),dst)
#
#                 # cv2.imshow('cam',cam)
#                 # cv2.imshow('dst',dst)
#                 # cv2.imshow('gray',gray)
#                 # cv2.imshow('roi',roi)
#                 # cv2.imshow('thresh1',thresh1)
#                 # cv2.imshow('th3',th3)
#                 temp_array = roi.reshape(1, 38400).astype(np.float32)
#                 frame += 1
#                 total_frame += 1
#
#                 if cv2.waitKey(1) & 0xFF == ord('l'):
#                     print("exit")
#
#                     break
#                 for event in pygame.event.get():
#                     if event.type == KEYDOWN:
#                         keys = pygame.key.get_pressed()
#                         if keys[pygame.K_w]:
#                             image_array = np.vstack((image_array, temp_array))
#                             label_array = np.vstack((label_array, self.k[0]))
#                             saved_frame += 1
#                             self.t_up(100, 100)
#                             cv2.imwrite('./training_images/{:1}_frame{:>05}.jpg'.format(label_0, saved_frame), th3)
#                             print("forward")
#                             print(saved_frame)
#
#                         elif keys[pygame.K_q]:
#                             image_array = np.vstack((image_array, temp_array))
#                             label_array = np.vstack((label_array, self.k[1]))
#                             saved_frame += 1
#                             self.t_up(50, 100)
#                             cv2.imwrite('./training_images/{:1}_frame{:>05}.jpg'.format(label_1, saved_frame), th3)
#                             print("left")
#                             print(saved_frame)
#
#                         elif keys[pygame.K_e]:
#                             image_array = np.vstack((image_array, temp_array))
#                             label_array = np.vstack((label_array, self.k[2]))
#                             saved_frame += 1
#                             self.t_up(100, 50)
#                             cv2.imwrite('./training_images/{:1}_frame{:>05}.jpg'.format(label_2, saved_frame), th3)
#                             print("right")
#                             print(saved_frame)
#
#                         elif keys[pygame.K_UP]:
#                             self.t_up(100, 100)
#                             print("forward")
#
#                         elif keys[pygame.K_s]:
#                             self.t_down(100, 100)
#                             print("back")
#                         elif keys[pygame.K_DOWN]:
#                             self.t_down(100, 100)
#                             print("back")
#
#                         elif keys[pygame.K_LEFT]:
#                             self.t_up(50, 100)
#                             print("left")
#
#                         elif keys[pygame.K_RIGHT]:
#                             self.t_up(100, 50)
#                             print("right")
#
#                         elif keys[pygame.K_a]:
#                             self.t_left(100, 100)
#                             print("turn left")
#
#                         elif keys[pygame.K_d]:
#                             self.t_right(100, 100)
#                             print("turn right")
#
#                         elif keys[pygame.K_z]:
#                             self.t_down(50, 100)
#                             print("left back")
#
#                         elif keys[pygame.K_c]:
#                             self.t_down(100, 50)
#                             print("right back")
#
#                         elif keys[pygame.K_p]:
#                             print('exit')
#                             self.save_img = False
#                             break
#
#                     elif event.type == KEYUP:
#                         self.t_stop(0)
#                         print("stop")
#
#         # save training images and labels
#
#         train = image_array[1:, :]
#         train_labels = label_array[1:, :]
#         # save training data as a numpy file
#
#         file_name = str(int(time.time()))
#         directory = "training_data"
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         try:
#             np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
#         except IOError as e:
#             print(e)
#         e2 = cv2.getTickCount()
#
#         time0 = (e2 - e1) / cv2.getTickFrequency()
#         print('Video duration:', time0)
#         print('Streaming duration:', time0)
#
#         print((train.shape))
#         print((train_labels.shape))
#         print('Total frame:', total_frame)
#         print('Saved frame:', saved_frame)
#         print('Dropped frame', total_frame - saved_frame)
#
#     finally:
#         self.cap.release()
#         cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     CollectData()
