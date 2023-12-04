import cv2
import numpy as np
import mediapipe as mp

# initialize mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)


def remove_background(file_person):
    person = cv2.imread(f"picture of person/{file_person}")
    person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)

    results = selfie_segmentation.process(person_rgb)
    mask = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5

    return mask
