#!/usr/bin/python
import os
import random
import shutil

shutil.rmtree("data20", ignore_errors=True)
os.makedirs("data20")
os.makedirs("data20/val")
os.makedirs("data20/train")
for f in random.sample(os.listdir("data/val"), 20):
    shutil.copytree("data/val/" + f, "data20/val/"+f)
    shutil.copytree("data/train/" + f, "data20/train/"+f)
