import cv2
import numpy as np

def freak_feature_matching(img1_path, img2_path):
    try:
        # Load images in grayscale
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            raise ValueError("One or both image paths are invalid or images cannot be loaded.")

        # Step 1: Detect keypoints (BRISK works well with FREAK)
        brisk = cv2.BRISK_create()
        kp1 = brisk.detect(img1, None)
        kp2 = brisk.detect(img2, None)

        # Step 2: Create FREAK descriptor
        freak = cv2.xfeatures2d.FREAK_create()

        # Step 3: Compute descriptors
        kp1, des1 = freak.compute(img1, kp1)
        kp2, des2 = freak.compute(img2, kp2)

        if des1 is None or des2 is None:
            raise ValueError("No descriptors found. Try different images or detector.")

        # Step 4: Match descriptors using Brute Force with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Step 5: Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Step 6: Draw matches
        result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

        # Display result
        cv2.imshow("FREAK Matches", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

# Example usage
freak_feature_matching("par11.jpg", "par12.jpg")
