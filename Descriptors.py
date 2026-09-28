import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

image1 = cv.imread("./img/par21.jpg")
image2 = cv.imread("./img/par22.jpg")

# ============================================================
# CONFIGURACIÓN FAST
# ============================================================

FAST_THRESHOLD = 20
FAST_NONMAX_SUPPRESSION = True
FAST_TYPE = cv.FAST_FEATURE_DETECTOR_TYPE_9_16


# ============================================================
# DETECTOR FAST
# ============================================================

def detect_fast(image):

    fast = cv.FastFeatureDetector_create(
        threshold=FAST_THRESHOLD,
        nonmaxSuppression=FAST_NONMAX_SUPPRESSION,
        type=FAST_TYPE
    )

    keypoints = fast.detect(image, None)

    return keypoints


# ============================================================
# DESCRIPTORES
# ============================================================

def create_descriptor(name):

    name = name.upper()

    if name == "ORB":

        return cv.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )

    elif name == "BRISK":

        return cv.BRISK_create(
            thresh=30,
            octaves=3,
            patternScale=1.0
        )

    elif name == "FREAK":

        return cv.xfeatures2d.FREAK_create(
            orientationNormalized=True,
            scaleNormalized=True,
            patternScale=22.0,
            nOctaves=4
        )

    elif name == "AKAZE":

        return cv.AKAZE_create(
            descriptor_type=cv.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.001,
            nOctaves=4,
            nOctaveLayers=4,
            diffusivity=cv.KAZE_DIFF_PM_G2
        )

    else:

        raise ValueError(
            f"Descriptor desconocido: {name}"
        )


# ============================================================
# DETECCIÓN + DESCRIPCIÓN
# ============================================================

def detect_and_describe(image, descriptor_name):

    #  FAST
    keypoints = detect_fast(image)

    # Descriptor seleccionado
    descriptor = create_descriptor(descriptor_name)

    if descriptor_name.upper() == "AKAZE":
        keypoints, descriptors = descriptor.detectAndCompute(
            image,
            None
        )

    else:

        # El descriptor utiliza  keypoints encontrados por FAST
        keypoints, descriptors = descriptor.compute(
            image,
            keypoints
        )

    return keypoints, descriptors


# ============================================================
# MATCHING
# ============================================================

def match_descriptors(desc1, desc2):

    if desc1 is None or desc2 is None:

        return []

    matcher = cv.BFMatcher(
        cv.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(
        desc1,
        desc2
    )

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    return matches


# ============================================================
# PROCESAMIENTO
# ============================================================

def process_images(image1, image2, descriptor_name):

    start = time.perf_counter()

    kp1, desc1 = detect_and_describe(
        image1,
        descriptor_name
    )

    kp2, desc2 = detect_and_describe(
        image2,
        descriptor_name
    )

    detection_time = time.perf_counter() - start

    matches = match_descriptors(
        desc1,
        desc2
    )

    return {
        "descriptor": descriptor_name,
        "keypoints_img1": len(kp1),
        "keypoints_img2": len(kp2),
        "descriptors_img1": 0 if desc1 is None else len(desc1),
        "descriptors_img2": 0 if desc2 is None else len(desc2),
        "matches": len(matches),
        "time": detection_time,
        "kp1": kp1,
        "kp2": kp2,
        "desc1": desc1,
        "desc2": desc2,
        "matches_data": matches
    }

descriptor = "FREAK"

resultado = process_images(
    image1,
    image2,
    descriptor
)

print("Detector: FAST")
print("Descriptor:", resultado["descriptor"])
print("Keypoints imagen 1:", resultado["keypoints_img1"])
print("Keypoints imagen 2:", resultado["keypoints_img2"])
print("Matches:", resultado["matches"])
print("Tiempo:", resultado["time"])

# ============================================================
# MOSTRAR MATCHES
# ============================================================

matches = resultado["matches_data"]

matched_image = cv.drawMatches(
    image1,
    resultado["kp1"],
    image2,
    resultado["kp2"],
    matches[:50],          # Mostrar los 50 mejores matches
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(16, 8))
plt.imshow(cv.cvtColor(matched_image, cv.COLOR_BGR2RGB))
plt.title(
    f"FAST + {resultado['descriptor']} - "
    f"{len(matches)} matches"
)
plt.axis("off")
plt.show()