#imports
import numpy as np
import cv2
import sys
import statistics as s
import math
import matplotlib.pyplot as plt


#helpers para o Gauss
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def amplifyGauss(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

def lstAvg(lst): 
    return sum(lst) / len(lst)   

def rmssd(lst):
	return  math.sqrt(lstAvg(lst))


#debug
debug = False

#face detection
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

# Webcam Parameters
realWidth = 512
realHeight = 512
videoWidth = 150
videoHeight = 100
videoChannels = 3
videoFrameRate = 15
video_capture.set(3, realWidth);
video_capture.set(4, realHeight);


#bpm arrays
bufferSize = 150
bufferIndex = 0
allBpmBuffer = []
snnSum = []


#Bases Para o Gaussian Pyramid / Fourier
levels = 3
alpha = 150
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

#Filtro para frequencias(1/2hz to 60-120bpm/m)
minFrequency = 1
maxFrequency = 2
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

#Bpm calculation vars
nIter = 5
bpmBufferIndex = 0
bpmBufferSize = 5
bpmBuffer = np.zeros((bpmBufferSize))

#Texto
font = cv2.FONT_HERSHEY_SIMPLEX 
bpmTextLocation = (0, 160)


while True:
	ret, frame = video_capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    #Desenhar o retangulo
	for (x, y, w, h) in faces:
		#h -- horizontal 
		#w -- vertical
		#c -- center
		#config rect para usar no gaus
		
		c = x + int(w/2)
		x += int((c-x)*0.4)
		h = videoHeight
		w = videoWidth

		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		detectionFrame = frame[y:y+h,x:x+w];

		#Gauss/Fourir para cada frame
		videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
		fourierTransform = np.fft.fft(videoGauss, axis=0)

		#filtrar frequencias
		fourierTransform[mask == False] = 0

		#Fazer medição de n em n iterecoes
		if bufferIndex % nIter == 0:
			for buf in range(bufferSize):
				fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
				hz = frequencies[np.argmax(fourierTransformAvg)]
			bpm = 60.0 * hz
			bpmBuffer[bpmBufferIndex] = bpm
			#calculo para 
			if(len(allBpmBuffer) > 2):
				#print("sd " + str(s.stdev(allBpmBuffer)))
				snnSum += [s.stdev(allBpmBuffer)]
				#print("sd mean" + str(rmssd(snnSum)))
			bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

		#mostrar gráfico
		if bufferIndex % 3600 == 0 and debug:
			plt.plot(allBpmBuffer)
			plt.ylabel('some numbers')
			plt.show()
			
		#AmplifyGauss & mostar variação de cores
		filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
		filtered = filtered * alpha		

		filteredFrame = amplifyGauss(filtered, bufferIndex, levels)
		outputFrame = detectionFrame + filteredFrame
		outputFrame = cv2.convertScaleAbs(outputFrame)

		bufferIndex = (bufferIndex + 1) % bufferSize

		frame[0:h,0:w] = outputFrame
		
		#print(bpmBuffer)

		if(bpmBuffer[-1] == 0):
			cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, 1,(0,0,255), 2)
		else:	
			allBpmBuffer += [bpmBuffer.mean()]
			cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, 1,(0,255,0), 2)

    #Mostrar webcam
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break