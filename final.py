import cv2 as cv
import numpy as np
import random
# Set up training data


gap = 2
harris_block_size = 2
xy_depth = 1
k_size = 5
def generate_feature_vector(img, ij, sobelx = None, sobely = None, w_sobel = False):
	y, x = ij
	if(w_sobel == False):
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		sobelx = cv.Sobel(gray, cv.CV_16S, xy_depth, 0, ksize=k_size)
		sobely = cv.Sobel(gray, cv.CV_16S, 0, xy_depth, ksize=k_size)
	cropx = sobelx[y - gap : y + gap, x - gap : x + gap]
	cropy = sobely[y - gap : y + gap, x - gap : x + gap]
	serial = np.concatenate((cropx, cropy))
	serial = serial.flatten()
	
	return serial


def generate_labels(train_files, labels, data):
	x = 0
	corners = 0
	noncorners = 0
	for file in train_files:
		img = cv.imread(file)
		cv.imshow('Org', img)
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		sobelx = cv.Sobel(gray, cv.CV_16S, xy_depth, 0, ksize=k_size)
		sobely = cv.Sobel(gray, cv.CV_16S, 0, xy_depth, ksize=k_size)
		gray = np.float32(gray)
		dst = cv.cornerHarris(gray, harris_block_size, 3, 0.04)

		# Threshold for an optimal value, it may vary depending on the image.
		img_array = dst > 0.01 * dst.max()
		nimg_array = dst <= 0.01 * dst.max()
		Corners = np.asarray(np.where(img_array == True)).T
		np.random.shuffle(Corners)
		NonCorners = np.asarray(np.where(img_array == False)).T
		np.random.shuffle(NonCorners)
		
		Corners = Corners[:int(len(Corners) * 0.9)]
		NonCorners = NonCorners[:int(len(NonCorners) * 0.9)]
		for y, x in Corners:
			if(y > gap and y < img_array.shape[0] - gap and x > gap and x < img_array.shape[1] - gap):
				labels.append(1)
				corners += 1
				data.append(generate_feature_vector(None, (y, x), sobelx=sobelx, sobely=sobely, w_sobel=True))
			
		for y, x in NonCorners:
			if(y > gap and y < img_array.shape[0] - gap and x > gap and x < img_array.shape[1] - gap):
				labels.append(-1)
				noncorners += 1
				data.append(generate_feature_vector(None, (y, x), sobelx=sobelx, sobely=sobely, w_sobel=True))
		
		img[img_array] = [255, 0, 0]
		img[nimg_array] = [0, 0, 255]

		cv.imshow("GT", img)
		q = cv.waitKey(1)
		if( q == 27 ):
			break
		elif( q == 32 ):
			cv.waitKey()

	print("Corners: ", corners, ", NonCorners: ", noncorners)
	return

def calc_SVM(svm, image):
	cv.imshow('Org', image)
	image1 = image.copy()
	gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
	sobelx = cv.Sobel(gray, cv.CV_16S, xy_depth, 0, ksize=k_size)
	sobely = cv.Sobel(gray, cv.CV_16S, 0, xy_depth, ksize=k_size)
	
	gray = np.float32(gray)
	dst = cv.cornerHarris(gray, harris_block_size, 3, 0.04)
	# Threshold for an optimal value, it may vary depending on the image.
	image1[dst > 0.01 * dst.max()] = [255, 0, 0]
	image1[dst < 0.01 * dst.max()] = [0, 0, 255]
	cv.imshow("GT", image1)
	
	# Show the decision regions given by the SVM
	for y in range(gap, image.shape[0] - gap):
		for x in range(gap, image.shape[1] - gap):
			sampleMat = np.matrix([generate_feature_vector(None, (y, x), sobelx = sobelx, sobely = sobely, w_sobel = True)], dtype=np.float32)
			response = svm.predict(sampleMat)[1]
			if response == 1:
				image[y,x] = (255, 0, 0)
			else:
				image[y,x] = (0, 0, 255)
	cv.imshow('Result', image) # show it to the user
	q = cv.waitKey(100)
	if( q == 32 ):
		cv.waitKey()
	return





if __name__ == "__main__":
	cv.namedWindow('Result',cv.WINDOW_NORMAL)
	cv.resizeWindow('Result', 600,600)
	cv.namedWindow('GT',cv.WINDOW_NORMAL)
	cv.resizeWindow('GT', 600,600)
	cv.namedWindow('Org',cv.WINDOW_NORMAL)
	cv.resizeWindow('Org', 600,600)
	
	import glob, os
	dir_files = glob.glob("train/*.jpg")
	dir_files = dir_files[:80]
	percantage = int(len(dir_files) * 0.8)
	train_images	= dir_files[:percantage]
	test_images		= dir_files[percantage +1:]
	
	labels = []
	trainingData = []
	generate_labels(train_images, labels, trainingData)
	print("Training with : ", len(labels), "samples")
	labels = np.array(labels)
	trainingData = np.matrix(trainingData, dtype=np.float32)

	# Train the SVM
	svm = cv.ml.SVM_create()
	svm.setType(cv.ml.SVM_C_SVC)
	svm.setKernel(cv.ml.SVM_LINEAR)
	svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
	svm.train(trainingData, cv.ml.ROW_SAMPLE, labels)

	for file in test_images:
		calc_SVM(svm, cv.imread(file))

