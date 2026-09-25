# Source - https://stackoverflow.com/a/65863713
# Posted by whoisraibolt, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-24, License - CC BY-SA 4.0

# Imports
import cv2 as cv



import matplotlib.pyplot as plt

# Open and convert the input and training-set image from BGR to GRAYSCALE
image1 = cv.imread(filename = 'par11.jpg',
                   flags = cv.IMREAD_GRAYSCALE)

image2 = cv.imread(filename = 'par12.jpg',
                   flags = cv.IMREAD_GRAYSCALE)


# Source - https://stackoverflow.com/a/65863713
# Posted by whoisraibolt, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-24, License - CC BY-SA 4.0

# Initiate BRISK descriptor
BRISK = cv.BRISK_create()

# Find the keypoints and compute the descriptors for input and training-set image
keypoints1, descriptors1 = BRISK.detectAndCompute(image1, None)
keypoints2, descriptors2 = BRISK.detectAndCompute(image2, None)


# Source - https://stackoverflow.com/a/65863713
# Posted by whoisraibolt, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-24, License - CC BY-SA 4.0

# create BFMatcher object
BFMatcher = cv.BFMatcher(normType = cv.NORM_HAMMING,
                         crossCheck = True)

# Matching descriptor vectors using Brute Force Matcher
matches = BFMatcher.match(queryDescriptors = descriptors1,
                          trainDescriptors = descriptors2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x: x.distance)

# Draw first 15 matches
output = cv.drawMatches(img1 = image1,
                        keypoints1 = keypoints1,
                        img2 = image2,
                        keypoints2 = keypoints2,
                        matches1to2 = matches[:15],
                        outImg = None,
                        flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(output)
plt.show()
