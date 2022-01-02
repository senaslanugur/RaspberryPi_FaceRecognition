"""
Melike Kubra Ozdemir - 12.2020

Yuz Maskesi Tanimlama Projesi

## Kaynak - https://github.com/EdjeElectronics ##


"""

import numpy as np
import sys
import time
import os
import argparse
import cv2
from threading import Thread
import importlib.util

# Video olaylarini tanimlamak icin, durdurma, baslatma ve yenileme gibi 
# Kaynak - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """PiCamera kullanamak icin kaynak kod"""
    def __init__(self,resolution=(640,480),framerate=30):
        # PiCamera baslangıcı
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # akıs esnasında ilk karenin okunmasi
        (self.grabbed, self.frame) = self.stream.read()

	# Kamera durduruldugunda gerekn degisken
        self.stopped = False

    def start(self):
	# Video baslagincinda akisi baslatmak icin
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # video durudurulana kadar surekli update eden kisim
        while True:
            # kamera durduruldugunda yapilacak islem
            if self.stopped:
                # kamerayi durdurmak
                self.stream.release()
                return

            # herhangi bir durumdan sonra tekrar devam edecegi kisim
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# en son kareyi dondrumesi icin
        return self.frame

    def stop(self):
	# kemera ve dongunun bitme durumu
        self.stopped = True

# bagimsiz degistenleri tanimlama ve parse etme islemleri
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_yolu', help='.Tflite dosyanin bulundugu path/yol',
                    required=True)
parser.add_argument('--dataset_adi_farkli', help=' Detect.tflite dosyasindan farkili ise .tflite dosyasinin adi',
                    default='detect.tflite')
parser.add_argument('--etiketler', help='Labelmap.txt den farkliysa, etiket esleme dosyasinin adi',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Algilanan nesneleri goruntulemek icin minimum guven esigi',
                    default=0.5)
parser.add_argument('--resolution', help='WxH de istenen web kamerasi cozunurlugu. Web kamerasi girilen cozunurlugu desteklemiyorsa hatalar meydana gelebilir.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Algılamayı hızlandırmak için Coral Edge TPU Acceleratori kullanin',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.dataset_yolu
GRAPH_NAME = args.dataset_adi_farkli
LABELMAP_NAME = args.etiketler
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu


pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate


if use_TPU:
    
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       


CWD_PATH = os.getcwd()


PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)


PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)


with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


# ilk satirini siliyor
if labels[0] == '???':
    del(labels[0])


if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5


frame_rate_calc = 1
freq = cv2.getTickFrequency()


videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)


while True:


    t1 = cv2.getTickCount()


    frame1 = videostream.read()


    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)


    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std


    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()


    boxes = interpreter.get_tensor(output_details[0]['index'])[0] 
    classes = interpreter.get_tensor(output_details[1]['index'])[0] 
    scores = interpreter.get_tensor(output_details[2]['index'])[0] 



    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):


            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)


            object_name = labels[int(classes[i])] 
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 


    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)


    cv2.imshow('Yuz Maskesi Tanimlama Projesi - Melike Kubra Ozdemir', frame)


    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1


    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
videostream.stop()