from run import process
import time
import subprocess
import os
import argparse
import cv2
import sys
from PIL import Image
import torch
import gradio as gr


TESTdevice = "cpu"

index = 1


"""
main.py

 How to run:
 python main.py

"""


def mainTest(inputpath, outpath):
    watermark = deep_nude_process(inputpath)
    cv2.imwrite(outpath, watermark)
    return watermark
    #


def deep_nude_process(item):
    # print('Processing {}'.format(item))
    # dress = cv2.imread(item)
    dress = (item)
    h = dress.shape[0]
    w = dress.shape[1]
    dress = cv2.resize(dress, (512, 512), interpolation=cv2.INTER_CUBIC)
    watermark = process(dress)
    watermark = cv2.resize(watermark, (w, h), interpolation=cv2.INTER_CUBIC)
    return watermark


def inference(img):
    global index
    # inputpath = "input/" + str(index) + ".jpg"
    outputpath = "out_" + str(index) + ".jpg"
    # cv2.imwrite(inputpath, img)
    index += 1
    output = mainTest(img, outputpath)
    return output


title = "XXXXXXXXXX"
description = "传入人物照片,类似最下方测试图的那种,将制作XX图,一张图至少等30秒"

examples = [
    ['input.png', '测试图'],
]


web = gr.Interface(inference,
                   inputs="image",
                   outputs="image",
                   title=title,
                   description=description,
                   examples=examples,
                   )

if __name__ == '__main__':
    web.launch(
        share=True,
        enable_queue=True
    )
