# import  sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import math
import cv2
import numpy as np

class Gripper:
    def __init__(self, center, angle, width, rel, base, max_val):
        self.x = center[0]
        self.y = center[1]
        self.angle = angle # in degree
        self.base_width = width
        self.rectangles = []
        self.max_x, self.max_y = max_val
        for x, y, w, h, angle in rel:
            # x, y should be relative center position, angle is relative to the angle
            el = Rectangle(x, y, w, h, angle)
            self.rectangles.append(el)
        self.base = Rectangle(base[0], base[1], base[2], base[3], base[4])
    
    def get_info(self):
        return [self.x, self.y, self.angle]
    
    def change_width(self, width):
        # print('self.base_width',self.base_width)
        if self.base_width == 0:
            self.base_width = 0.001
        factor = width / self.base_width
        #factor = 1
        
        self.base_width = width
        for rect in self.rectangles:
            rect.x = rect.x * factor
            # rect.y = rect.y * factor
        return factor
    
    def change_width_dw(self, dw):
        width = self.base_width + dw
        if width >= self.base.w:
            width = self.base.w - 1
        elif width < 1:
            width = 1
        factor = width / self.base_width
        self.base_width = width
        for rect in self.rectangles:
            rect.x = rect.x * factor
            # rect.y = rect.y * factor
        return factor

    def translate(self, dx, dy):
        self.x += dx
        if self.x < 0:
            self.x = 0
        elif self.x >= self.max_x:
            self.x = self.max_x - 1
        self.y += dy
        if self.y < 0:
            self.y = 0
        elif self.y >= self.max_y:
            self.y = self.max_y - 1
    
    def rotate(self, angle):
        self.angle += angle
    
    def get_contours(self, base=False):
        contours = []
        if base:
            rect = self.base
            angle = math.pi * (self.angle / 180)
            center = np.array([rect.x * math.cos(angle) - rect.y * math.sin(angle) + self.x, rect.x * math.sin(angle) + rect.y * math.cos(angle) + self.y])
            box = [center, (rect.w, rect.h), rect.angle + self.angle]
            points = cv2.boxPoints(box)
            box = np.int0(points)
            contours.append(box)
        else:
            for rect in self.rectangles:
                angle = math.pi * (self.angle / 180)
                center = np.array([rect.x * math.cos(angle) - rect.y * math.sin(angle) + self.x, rect.x * math.sin(angle) + rect.y * math.cos(angle) + self.y])
                box = [center, (rect.w, rect.h), rect.angle + self.angle]
                points = cv2.boxPoints(box)
                box = np.int0(points)
                contours.append(box)
        
        # just for checking center, need to comment this part
        # box = [np.array([self.x, self.y]), (1, 1), 0]
        # points = cv2.boxPoints(box)
        # box = np.int0(points)
        # contours.append(box)

        return contours

    def check_in(self):
        all_in = True
        for rect in self.rectangles:
            angle = math.pi * (self.angle / 180)
            center = np.array([rect.x * math.cos(angle) - rect.y * math.sin(angle) + self.x, rect.x * math.sin(angle) + rect.y * math.cos(angle) + self.y])
            box = [center, (rect.w, rect.h), rect.angle + self.angle]
            points = np.int0(cv2.boxPoints(box)).tolist()
            for x, y in points:
                if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
                    all_in = False
                    break
            if not all_in:
                break
        return all_in

class Rectangle:
    def __init__(self, x, y, w, h, angle):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle # in degree

# https://github.com/rij12/YOPO/blob/yopo/darkflow/net/yopo/calulating_IOU.py
