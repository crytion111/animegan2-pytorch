# coding=utf-8
import sys
import cv2
import argparse
import os
import sys
import subprocess
import time

from run import process

"""
main.py

 How to run:
 python main.py

"""


def main(inputpath, outpath, show):
	if isinstance(inputpath, list):
		for item in inputpath:
			watermark = deep_nude_process(item)
			cv2.imwrite("output_"+item, watermark)
	else:
		watermark = deep_nude_process(inputpath)
		cv2.imwrite(outpath, watermark)

def deep_nude_process(item):
    print('Processing {}'.format(item))
    dress = cv2.imread(item)
    h = dress.shape[0]
    w = dress.shape[1]
    dress = cv2.resize(dress, (512,512), interpolation=cv2.INTER_CUBIC)
    watermark = process(dress)
    watermark = cv2.resize(watermark, (w,h), interpolation=cv2.INTER_CUBIC)
    return watermark

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="simple deep nude script tool")
	parser.add_argument("-i", "--input", action="store", nargs = "*", default="input.png", help = "Use to enter input one or more files's name")
	parser.add_argument("-o", "--output", action="store", default="output.png", help = "Use to enter output file name")
	parser.add_argument("-s", "--show", action="store", default="false", help = "Use to automatically display or not display generated images")
	inputpath, outputpath, show = parser.parse_args().input, parser.parse_args().output, parser.parse_args().show
    
	print (time.strftime("START!!!!!!!!! %Y-%m-%d %H:%M:%S", time.localtime()))
	main(inputpath, outputpath, show)
	print (time.strftime("FINISH!!!!!!!!! %Y-%m-%d %H:%M:%S", time.localtime()))
