import cv2
import sys
import numpy as np

confThreshold = 0.3
nmsThreshold = 0.2

def get_outputs_names(net):
	layersNames = net.getLayerNames()
	return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom):
	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

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

	indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		i = i[0]
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

if __name__ == '__main__':

	if (len(sys.argv) < 2):
		cap = cv2.VideoCapture(0)
	else:
		cap = cv2.VideoCapture(sys.argv[1])

	classesFile = "darknet/cfg/coco.data";
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	modelConfiguration = "yolo/yolov3-tiny.cfg";
	modelWeights = "yolo/yolov3-tiny.weights";

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
		blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(get_outputs_names(net))

		postprocess(frame, outs)

		cv2.imshow('YoloV3', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	exit()
	