#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:49:00 2021

@author: isabel
"""
hw_string = "Hello world"

with open("hello_world.txt", "w") as text_file:
    text_file.write(hw_string)
