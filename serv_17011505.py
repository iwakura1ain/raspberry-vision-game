#!/usr/bin/python3
"""
Embedded System Project
17011505 An ChangErn
"""

import curses
from curses import wrapper

from time import sleep
from random import randint, seed
import datetime
import logging as l

from sense_hat import SenseHat
import serial 

OBSTACLE_CNT = 1
SENSE_COLORS = {
    "0": (0, 0, 0),  # nothing 
    "=": (255, 0, 0),  # obstacle  
    "^": (255, 255, 255)  # ship
}

#l.basicConfig(filename="test.log", encoding="utf=8", level=l.DEBUG)

class Display:
    def __init__(self, **kwargs):
        self.arr = [["0" for x in range(0, 8)] for y in range(0, 8)]
        
        if kwargs["type"] == "curses":
            self.stdscr = kwargs["stdscr"]
        elif kwargs["type"] == "sense":
            self.sense = SenseHat()
        elif kwargs["type"] == "both":
            self.stdscr = kwargs["stdscr"]
            self.sense = SenseHat()
            
    def set_arr(self, ship, obstacles):
        self.arr = [["0" for x in range(0, 8)] for y in range(0, 8)].copy()
        #        l.debug(f'{}ship.current[1], ship.current[0])

        if not ship.is_dead:
            self.arr[ship.current[1]][ship.current[0]] = "^"
            
        for obs in obstacles:
            for obs_x in range(obs.current[0], obs.current[1]):
                self.arr[obs_x][obs.height] = "="
        
    def print_curses(self):
        try:
            #self.stdscr.erase()
            for x, row in enumerate(self.arr):
                for y, val in enumerate(row):
                    self.stdscr.addstr(y, x, val)
                    self.stdscr.clrtobot()                    

            self.stdscr.refresh()
                    
        except NameError:
            pass
        
    def print_sense(self):
        try:
            for y, row in enumerate(self.arr):
                for x, val in enumerate(row):
                    self.sense.set_pixel(7-x, 7-y, SENSE_COLORS[val])
                    
        except NameError:
            pass
        
    def print_input(self, action):
        try:
            if action is not None:
                self.stdscr.addstr(9, 0, "Serial input: " + action.decode())
                self.stdscr.refresh()
        except:
            pass
        
    def wait_input(self):
        self.stdscr.nodelay(False)
        self.stdscr.getch()
        self.stdscr.nodelay(True)
        
class Ship:
    def __init__(self):
        self.current = [7, 3]
        self.is_dead = False
        
    def move_curses(self, input):
        if input == curses.KEY_LEFT and self.current[1] < 7:
            self.current[1] += 1
        elif input == curses.KEY_RIGHT and self.current[1] > 0:
            self.current[1] -= 1

    def move_serial(self, input):
        try:
            if input is not None:
                self.current[1] += int(input)

        except ValueError:
            pass

    def check_collision(self, obstacles):
        for obs in obstacles:
            for i in range(obs.current[0], obs.current[1]):
                if self.current == [obs.height+1, i]:
                    self.is_dead = True
                    return True

        self.is_dead = False
        return False

class Input:
    def __init__(self, **kwargs):
        self.type = kwargs["type"]
        
        if self.type == "curses":
            self.source = kwargs["stdscr"]
        elif self.type == "serial":
            self.ser = serial.Serial("/dev/serial0", 9600)
        elif self.type == "both":
            self.source = kwargs["stdscr"]
            self.ser = serial.Serial("/dev/serial0", 9600)

    def get_input(self):
        if self.type in ("curses", "both"):
            #return self.get_input_curses()
            return self.get_input_serial()
        elif self.type == "serial":
            return self.get_input_serial()

        
    def get_input_curses(self):
        return self.source.getch()

    def get_input_serial(self):
        if self.ser.in_waiting > 0:
            return self.ser.read(size=2)
        else:
            return None 



class Obstacle:
    def __init__(self, turn):
        self.turn = turn

        
        self.height = 0
        self.current = [0, 0]
        self.reset_obstacle(4)

    def move_curses(self, turn, current_ship):
        if self.turn == turn: 
            self.height = (self.height + 1) % 8 
            if self.height == 7:
                self.reset_obstacle(current_ship)
                
    def reset_obstacle(self, current_ship):
        self.height = 0

        if current_ship <= 3:
            self.current = [0, randint(4, 7)]
        elif current_ship >= 4:
            self.current = [randint(1, 3), 8]
        
def initialize_game(stdscr):
    curses.curs_set(False)
    curses.start_color()
    stdscr.nodelay(True)
    seed(datetime.datetime.now())
    
    return Ship(), Display(type="both", stdscr=stdscr), Input(type="both", stdscr=stdscr), [Obstacle(i) for i in range(0, OBSTACLE_CNT)]


def main(stdscr):
    while True:
        turn = 0
        ship, display, input, obstacles = initialize_game(stdscr)
        while True:
            display.set_arr(ship, obstacles)
            display.print_curses()
            display.print_sense()
            
            #ch = input.get_input_curses()
            #ship.move_curses(ch)

            ch = input.get_input_serial()
            ship.move_serial(ch)
            display.print_input(ch)
            
            for obs in obstacles:
                obs.move_curses(turn, ship.current[1])
                
            if ship.check_collision(obstacles):
                display.set_arr(ship, obstacles)
                display.print_curses()
                display.print_sense()
                sleep(2)
                break

            
            turn = (turn + 1) % 6
            #if turn == 0:
            #    display.wait_input()
            sleep(0.03)

            

    exit(0)

    
if __name__=="__main__":
    wrapper(main)











