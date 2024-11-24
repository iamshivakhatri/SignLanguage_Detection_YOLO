# script using yolov5 to train the model
# Usage: python train.py --data data.yaml --cfg yolov5s.yaml --weights '' --batch-size 64 --epochs 300 --device 0 --name yolov5s_results --img-size 640 --cache

import argparse
import os
import sys
from pathlib import Path
