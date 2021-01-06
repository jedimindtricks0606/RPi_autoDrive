import cv2
import numpy as np
import math
import RPi.GPIO as GPIO
import time
import pygame

#先定义出方向控制的方法：
class Carctrl(object):

    def __init__(self):
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

        GPIO.setup(self.ENA, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.IN1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.ENB, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.IN3, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN4, GPIO.OUT, initial=GPIO.LOW)
        # 设置pwm引脚和频率为2000hz
        self.pwm_ENA = GPIO.PWM(self.ENA, 2000)
        self.pwm_ENB = GPIO.PWM(self.ENB, 2000)
        # 启动PWM设置占空比为100（0--100）
        self.pwm_ENA.start(0)
        self.pwm_ENB.start(0)

    # 小车前进
    def t_up(self, leftspeed, rightspeed):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(leftspeed)
        self.pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车后退
    def t_down(self, leftspeed, rightspeed):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.HIGH)
        self.pwm_ENA.ChangeDutyCycle(leftspeed)
        self.pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车左转
    def t_left(self, leftspeed, rightspeed):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(leftspeed)
        self.pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车右转
    def t_right(self, leftspeed, rightspeed):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(leftspeed)
        self.pwm_ENB.ChangeDutyCycle(rightspeed)

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
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(0)
        self.pwm_ENB.ChangeDutyCycle(0)

    #模仿手动操作的方法，将方向的控制权交给计算机：
    def self_driving(self, prediction):
        if prediction == 0:
            self.t_up(10, 10)
            print("Forward")
        elif prediction == 1:
            self.t_up(5, 10)
            print("Left")
        elif prediction == 2:
            self.t_up(10, 5)
            print("Right")
        else:
            self.t_stop(0)


#接下来读取模型，将之前训练好的xml文件加载到程序中：
class NeuralNetwork(object):

    def __init__(self):
        self.annmodel = cv2.ml.ANN_MLP_load('mlp_xml/mlp.xml')

    def predict(self, samples):
        ret, resp = self.annmodel.predict(samples)
        return resp.argmax(-1)  # find max


#现在模型有了，如何控制方向的方法也有了，接下来该打开我们的摄像头，让“机器”看到前方的道路。同样，初始化神经网络和摄像头参数：
class CamDataHandle(object):

    def __init__(self):

        self.model = NeuralNetwork()
        print('load ANN model.')
        #self.obj_detection = ObjectDetection()
        self.car = Carctrl()

        print('----------------Caminit completed-----------------')
        self.handle()

    def handle(self):

        self.cap = cv2.VideoCapture(1)
        self.cap.set(3, 320)
        self.cap.set(4, 240)

        #获取图像并处理：
        try:
            while True:
                ret, cam = self.cap.read()
                gray = cv2.cvtColor(cam, cv2.COLOR_RGB2GRAY)
                ret, th3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                roi = th3[120:240, :]

                image_array = roi.reshape(1, 38400).astype(np.float32)
                prediction = self.model.predict(image_array)
                # cv2.imshow('roi',roi)
                cv2.imshow('cam', cam)

                if cv2.waitKey(1) & 0xFF == ord('l'):
                    break
                else:
                    self.car.self_driving(prediction)

        finally:
            cv2.destroyAllWindows()
            print('Shut down')


if __name__ == '__main__':
    CamDataHandle()