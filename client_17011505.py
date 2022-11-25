#!/usr/bin/python3
"""
Embedded System Project
17011505 An ChangErn
"""

import numpy as np
import imutils
import cv2

from picamera import PiCamera
from picamera.array import PiRGBArray
import io

import serial

import curses
from curses import wrapper

import time
import logging as l

CURSES_CHARS = {
    255:"0",
    0:"X"
}

l.basicConfig(filename="client.log")

class ScanException(Exception):
    pass


class Camera:
    def __init__(self):
        self.cnt = 0
        
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 32
        time.sleep(0.1)

        self.rawcap = PiRGBArray(self.camera, size=(640, 480))

    def get_frame(self):
        for frame in self.camera.capture_continuous(self.rawcap, format="bgr", use_video_port=True):
            yield frame.array

            self.rawcap.truncate(0)
            self.rawcap.seek(0)
            self.cnt += 1


class Display:
    def __init__(self, stdscr):
        self.stdscr = stdscr

    def print_matrix(self, matrix, cnt):
        self.stdscr.erase()
        if matrix is None:
            for y in range(0, 8):
                for x in range(0, 8):              
                    self.stdscr.addstr(y, x, "N")
                    
            return
        else:
            for y, row in enumerate(matrix):
                for x, val in enumerate(row):              
                    self.stdscr.addstr(y, x, CURSES_CHARS[val])

        self.stdscr.clrtobot()
        self.stdscr.refresh()
        
    def print_movement(self, action_queue):
        if len(action_queue) > 0:
            self.stdscr.addstr(9, 0, str(action_queue))
            self.stdscr.refresh()


class ImageScanner:
    def __init__(self):
        self.the_orig = None
        self.orig = None
        self.edged = None
        self.ratio = None
        self.warp = None
        self.pixel = None

    def scan_frame(self, image, cnt):
        try:
            tmp1 = self.preprocess(image)
            tmp2 = self.dectect_contour(tmp1)
            tmp3 = self.find_contour(tmp2)
            tmp4 = self.warp_contour(tmp3)
            tmp5 = self.crop(tmp4)
            #cv2.imwrite(f"image/cropped{cnt}.png", tmp5)
            tmp6 = self.pixelate(tmp5)

            return tmp6

        except ScanException as e:
            #l.error(e)
            return None
        
    def preprocess(self, image):
        try:
            image = cv2.flip(image, 1)
            self.the_orig = image.copy()
            self.ratio = image.shape[0] / 900.0

            image = imutils.resize(image, height=900)
            image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
            image = cv2.blur(image, (5, 5))
            self.orig = image.copy()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            return image

        except:
            raise ScanException("preprocessing error ")
        
    def dectect_contour(self, image):
        try:
            image = cv2.bilateralFilter(image, 11, 17, 17)
            edged = cv2.Canny(image, 30, 200)
            self.edged = edged
            
            cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[:10]

            return cnts
        
        except:
            raise ScanException("contour dectection error")
        
    def find_contour(self, cnts):
        try:
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.075 * peri, True)

                if len(approx) == 4:
                    return approx
        except:
            raise ScanException("contour location error")
        
    def warp_contour(self, screenCnt):
        try:
            pts = screenCnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            rect *= self.ratio

            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(self.the_orig, M, (maxWidth, maxHeight))

            return warp

        except:
            raise ScanException("perspective warp error")
        
    def crop(self, warp, h_start=19, h_end=20, w_start=20, w_end=18):
        try:
            (h, w) = warp.shape[:2]
            warp = imutils.rotate(warp, -0.4)
            warp = warp[h_start : h-h_end, w_start : w-w_end]
            self.warp = warp

            return warp
        
        except:
            raise ScanException("crop error")
        
    def pixelate(self, warp):
        try:
            pixel = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            pixel = cv2.resize(pixel, (8, 8), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            (thresh, pixel) = cv2.threshold(pixel, 250, 255, cv2.THRESH_BINARY_INV)
            self.pixel = pixel
            
            return pixel

        except:
            raise ScanException("pixelization error")



                                
class Actor:
    def __init__(self):
        self.prev_matrix = [[0 for x in range(0, 8)] for y in range(0, 8)]
        self.prev_ship_pos = 0
        
        self.curr_matrix = [[0 for x in range(0, 8)] for y in range(0, 8)]
        self.curr_ship_pos = 4

        self.action_queue = []
        self.queue_len = 0

    def locate_elements(self, scanned):
        self.prev_matrix = self.curr_matrix.copy()
        self.prev_ship_pos = self.curr_ship_pos
        
        self.curr_matrix = scanned.copy()
        self.curr_ship_pos = np.argmin(scanned[7]) #scanned[7].index(0)

        if self.curr_ship_pos == -1:
            self.pass_turn()
            return False 
        elif not self.verify_action():
            self.pass_turn()
            return False
        else:
            return True
        

    def get_path(self, x):
        len = 0
        for h in range(6, -1, -1):
            if self.curr_matrix[h][x] != 0:
                len += 1
            else:
                break

        return len

    def get_action(self):
        r_max = [0, 0]
        for x in range(self.curr_ship_pos, 8):
            if r_max[1] < (tmp_max := self.get_path(x)):
                r_max = (x - self.curr_ship_pos, tmp_max)

        l_max = [0, 0]
        for x in range(self.curr_ship_pos, -1, -1):
            if l_max[1] < (tmp_max := self.get_path(x)):
                l_max = (x - self.curr_ship_pos, tmp_max)
                
        if l_max[1] < r_max[1]:
            return r_max[0]
        else:
            return l_max[0]

    def queue_action(self, offset):
        if offset == 0:
            self.action_queue.clear()
        elif offset < 0:
            self.action_queue.extend(["-1" for i in range(0, abs(offset))])
        elif offset > 0:
            self.action_queue.extend(["+1" for i in range(0, abs(offset))])

        return self.action_queue

    def verify_action(self):
        offset = abs(self.prev_ship_pos - self.curr_ship_pos)

        for i in range(0, offset):
            try:
                self.action_queue.pop(0)
            except IndexError:
                return True
            
        if len(self.action_queue) > 0:
            if len(self.action_queue) == self.queue_len:
                self.action_queue.clear()
                self.queue_len = 0

                return True

            else:
                return False
        else:
            return True

    def pass_turn(self):
        self.curr_ship_pos = self.prev_ship_pos
        self.curr_matrix = self.prev_matrix
        #self.action_queue.clear()

class Output:
    def __init__(self):
        self.output = serial.Serial("/dev/ttyS0", 9600)

    def send_action(self, action_queue):
        for a in action_queue:
            self.output.write(a.encode('utf-8'))


def initialize_client(stdscr):
    return Display(stdscr), ImageScanner(), Camera(), Actor(), Output()


def main(stdscr):
    display, scanner, camera, actor, output = initialize_client(stdscr)

    cnt = 0
    for frame in camera.get_frame():
        #cv2.imwrite(f"image/capture{cnt}.png", frame)
        
        if (scanned := scanner.scan_frame(frame, cnt)) is not None:
            display.print_matrix(scanned, cnt)
            #cv2.imwrite(f"image/scan_succees{cnt}.png", scanned)

            if actor.locate_elements(scanned):
                l.error(f'ship index: {actor.curr_ship_pos}')
                
                new_actions = actor.queue_action(actor.get_action())
                display.print_movement(new_actions)
                output.send_action(new_actions)

            #cv2.imwrite(f"image/scan_fail{cnt}.png", scanned)
            #display.print_matrix(None, cnt)
            #l.error(f"frame {camera.cnt} unrecognized")

        else:
            display.print_matrix(None, cnt)

        if camera.cnt > 300:
            break

        cnt += 1

if __name__=="__main__":
    wrapper(main)


    #main()

















