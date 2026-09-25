import numpy as np
import cv2
import matplotlib.pyplot as plt
query_img = cv2.imread('par11.jpg')
train_img = cv2.imread('par12.jpg')

query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()

queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)
final_img = cv2.drawMatches(query_img, queryKeypoints, 
                             train_img, trainKeypoints, matches[:20], None)
final_img = cv2.resize(final_img, (1000, 650))


plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)) 
plt.title("Feature Matches")
plt.axis('off')  
plt.show()