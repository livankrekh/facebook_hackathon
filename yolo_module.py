import cv2
import sys
import os
import numpy as np
from gtts import gTTS
import speak_module
import threading

confThreshold = 0.4
nmsThreshold = 0.3
classes = []
detectedClasses = []

def get_outputs_names(net):
	layersNames = net.getLayerNames()
	return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(frame, classId, conf, left, top, right, bottom):
	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)

	label = '%.2f' % conf

	if classes:
		assert(classId < len(classes))
		label = '%s:%s' % (classes[classId], label)

	labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	top = max(top, labelSize[1])
	cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

def postprocess(frame, outs):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]

	classIds = []
	confidences = []
	boxes = []
	classIds = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confThreshold:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])

	global detectedClasses
	detectedClasses = []
	indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		i = i[0]
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		iheight, iwidth, _ = frame.shape
		if (iheight * iwidth * 0.75 <= width * height):
			tts = gTTS(text=str("Осторожно! Вы очень близко к некому обьекту"), lang='ru')
			tts.save('speech.mp3')
			os.system('mpg321 speech.mp3')
		drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
		detectedClasses.append(classes[classIds[i]])

def checkCapture(detectedArr):
	res = 0
	last = detectedArr[len(detectedArr) - 1]

	if (speak_module.result_rec.find("Погода") != -1 or speak_module.result_rec.find("погода") != -1):
		tts = gTTS(text=str("В городе Киев сегодня будет солнечная погода с переменной облачностью"), lang='ru')
		tts.save('speech.mp3')
		os.system('mpg321 speech.mp3')
		speak_module.result_rec = ""

	if (speak_module.result_rec.find("где я") != -1 or speak_module.result_rec.find("Где я") != -1):
		tts = gTTS(text=str("К сожалению, не могу определить ваше местоположение"), lang='ru')
		tts.save('speech.mp3')
		os.system('mpg321 speech.mp3')
		speak_module.result_rec = ""

	if (speak_module.result_rec.find("сколько автомобилей") != -1
		or speak_module.result_rec.find("сколько машин") != -1):

		tts = gTTS(text=str("Я вижу " + str(last.count('car')) + " машин"), lang='ru')
		tts.save('speech.mp3')
		os.system('mpg321 speech.mp3')
		speak_module.result_rec = ""

	if (speak_module.result_rec.find("посчитай людей") != -1
		or speak_module.result_rec.find("ты видишь") != -1
		or speak_module.result_rec.find("сколько людей") != -1):
		tts = gTTS(text=str("Я вижу " + str(last.count('person')) + " человека"), lang='ru')
		tts.save('speech.mp3')
		os.system('mpg321 speech.mp3')
		speak_module.result_rec = ""

	if (speak_module.result_rec.find("впереди") != -1):
		if (len(last) != 0):
			tts = gTTS(text=str("Впереди что-то есть. Это " + str(last[0]) ), lang='ru')
		else:
			tts = gTTS(text=str("Впереди ничего нет"), lang='ru')
		tts = gTTS(text=str("Я вижу " + str(last.count('person')) + " человека"), lang='ru')
		tts.save('speech.mp3')
		os.system('mpg321 speech.mp3')
		speak_module.result_rec = ""

	if (len(detectedArr) >= 100):
		for det in detectedArr:
			if (det.count('person') == 1):
				res += 1
		detectedArr.clear()

def start_detection(argv):

	if (len(argv) < 2):
		cap = cv2.VideoCapture(0)
	else:
		cap = cv2.VideoCapture(argv[1])

	t1 = threading.Thread(target=speak_module.main)
	t1.start()

	classesFile = "yolo/coco.names";
	global classes
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	modelConfiguration = "yolo/yolov3-tiny.cfg";
	modelWeights = "yolo/yolov3-tiny.weights";
	captureIter = []

	net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

	while(True):
		hasFrame, frame = cap.read()
		grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if not hasFrame:
			print("Done processing !!!")
			print("Output file is stored as ", outputFile)
			exit()

		imHeight, imWidth, _ = frame.shape
		blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(get_outputs_names(net))

		postprocess(frame, outs)
		captureIter.append(detectedClasses[:])
		checkCapture(captureIter)

		cv2.imshow('YoloV3', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
