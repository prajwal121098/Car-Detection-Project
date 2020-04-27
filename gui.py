# python gui.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel


import tkinter as tk, threading
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import numpy as np
from PIL import ImageTk, Image
import imageio
from pathlib import Path
import shutil
import os
import argparse
import pyttsx3
#import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


window = tk.Tk()
window.title("Car Detection")

window.geometry("500x580")

window.configure(background ="lightblue")

title = tk.Label(text="CAR DETECTION PROJECT", background = "lightblue", fg="Brown", font=("", 15))
title.grid(padx = 120, pady = 60)

#style = ttk.Style()
#style.configure('TButton', font = ('calibri', 10, 'bold', 'underline'), foreground = 'red')

# class liveStream(object):
# 	panel = None
# 	window1 = None
# 	camera = None
	
# 	def __init__(self):
# 		window.destroy()
# 		self.window1 = tk.Tk()
# 		self.window1.title('Car Detection - Video Stream')
# 		self.window1.geometry("400x400")
# 		#initalize a panel
# 		self.panel = tk.Label()
# 		#initalize camera..
# 		self.camera = cv2.VideoCapture(0)
# 		self.camera1()
# 		self.window1.mainloop()

# 	def camera1(self):
# 		ret, frame = self.camera.read()
		
# 		if(type(frame) == type(None)):
# 			exit()

# 		self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 		self.frame = Image.fromarray(frame)
# 		self.frame = ImageTk.PhotoImage(frame)
# 		if self.panel is None:
# 			self.panel = tk.Label(image=frame)
# 			self.panel.image = frame
# 			self.panel.pack(side="left", padx=10, pady=10)
# 			#self.panel.after(1, self.camera1)
# 		else:
# 			self.panel.configure(image=frame)
# 			self.panel.image = frame
 

# 	def exit():
# 		window1.destroy()


def liveStream():

	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# initialize the video stream, allow the cammera sensor to warmup,
	# and initialize the FPS counter
	print("[INFO] starting video stream...")
	vs = VideoStream().start()
	time.sleep(2.0)
	fps = FPS().start()

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				if idx == 7:
					cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

		# show the output frame
		cv2.imshow("Live Stream", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
'''	import cv2
	car_cascade = cv2.CascadeClassifier("car/haarcascade_cars.xml")

	
	cap = cv2.VideoCapture(0)

	e = pyttsx3.init()

	while True:
	    ret, img = cap.read()
	    if (type(img) == type(None)):
	        break
	    
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    
	    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

	    for (x,y,w,h) in cars:
	        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	        #playsound('sound/beep.wav')   
	        #print("car is detected")
	        #e.say("car is detected")
	        #e.runAndWait()

	    cv2.imshow('video', img)
	    
	    if cv2.waitKey(33) & 0xFF == ord('q'):
	        break

	cv2.destroyAllWindows()
'''

def videoStream():
	import cv2

	msg = messagebox.showinfo("Information","choose mv4 file for detection of car...")
	
	if msg:
		fileName = askopenfilename(initialdir='C:/Users/Prajwal/Desktop/Project', title='Select video for detection ', filetypes=[('video files', '.avi'),('video files','.mp4')])
	dst = "C:/Users/Prajwal/Desktop/Project/testvideo"
	shutil.copy(fileName, dst)
	
	car_cascade = cv2.CascadeClassifier("car/haarcascade_cars.xml")
	video_src = os.path.basename(fileName)
	cap = cv2.VideoCapture(video_src)
	e = pyttsx3.init()

	#window2 = tk.Tk()
	#window2.title('Car Detection - Video')
	#window2.geometry("800x480")
	#window2.configure(background="lightblue")
	#window2.resizable(0,0)

	#lmain = tk.Label(window2)
	#lmain.grid(row=0, column=0, padx=10, pady=10)
	
	while(cap.isOpened()):
	    ret, frame = cap.read()
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    
	    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

	    label = "Car"
	    for (x,y,w,h) in cars:
	        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
	        y = y - 15 if y - 15 > 15 else y + 15
	        cv2.putText(frame, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 2)
	        #playsound('sound/beep.wav')   
	        #print("car is detected")
	        #e.say("car is detected")
	        #e.runAndWait()

	    cv2.imshow('Video Stream', frame)
	    
	    if cv2.waitKey(33) & 0xFF == ord('q'):
	        break

	cv2.destroyAllWindows()


class MyDialog:
    def __init__(self):
    	self.window3 = tk.Tk()
    	self.window3.title('IP Value')
    	self.window3.geometry("200x100")
    	self.label = tk.Label(self.window3, text="Enter IP Address of Camera").pack()
    	self.exlabel = tk.Label(self.window3, text="Ex: https://192.168.23.1:8080").pack() 
    	self.e = tk.Entry(self.window3)
    	self.e.pack(padx=5)
    	self.b= tk.Button(self.window3, text="OK", command=self.ok)
    	self.b.pack(pady=5)
        
    def ok(self):

        print("value is", self.e.get())
        self.url = self.e.get()
        self.url = self.url+'/video'
        self.window3.destroy()
        window.destroy()

        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        # load our serialized model from disk
        print('[INFO] Loading Model...')
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        
        # initialize the video stream, allow the cammera sensor to warmup,
		# and initialize the FPS counter
        print('[INFO] Starting Video Stream...')
        vs = VideoStream(self.url).start()
        time.sleep(2.0)
        fps = FPS().start()

        # loop over the frames from the video stream
        while True:
        	frame = vs.read()
        	frame = imutils.resize(frame, width=400)

        	(h, w) = frame.shape[:2]
        	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

        	net.setInput(blob)
        	detections = net.forward()

        	for i in np.arange(0, detections.shape[2]):

        		confidence = detections[0, 0, i, 2]
        		if confidence > args["confidence"]:
        			idx = int(detections[0, 0, i, 1])
        			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        			(startX, startY, endX, endY) = box.astype("int")

        			label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)

        			if idx == 7:
        				cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
        				y = startY - 15 if startY - 15 > 15 else startY + 15
        				cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        	cv2.imshow("Ip Camera Stream", frame)
        	key = cv2.waitKey(1) & 0xFF
        	if key == ord('q'):
        		break

        	fps.update()
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        cv2.destroyAllWindows()
        vs.stop()

def exit():
	window.destroy()

button = tk.Button(text="Live Detection", height=2, width=15, command= liveStream)
button.grid(column=0, row=1, padx=10, pady = 10)

button1 = tk.Button(text="Video Detection", height=2, width=15, command= videoStream)
button1.grid(column=0, row=2, padx=10, pady = 10)

button1 = tk.Button(text="IP Camera", height=2, width=15, command= MyDialog)
button1.grid(column=0, row=3, padx=10, pady = 10)

msg = tk.Label(window, text = "Note - Press 'q' to close streaming frame...")
msg.grid(column=0, row=4, padx=10, pady=10)

exit = tk.Button(text = "Exit", height=2, width=15, command= exit)
exit.grid(column=0, row=5, padx=10, pady=10)

window.mainloop()