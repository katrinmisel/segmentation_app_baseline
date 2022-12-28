import numpy as np
from keras.models import load_model
import cv2
import segmentation_models as sm

def getPrediction(filename):

    my_model = load_model("unet_resnet34_augmented.h5", custom_objects={'focal_loss_plus_dice_loss': sm.losses.categorical_focal_dice_loss})

    original_image = cv2.resize(cv2.cvtColor(cv2.imread('static/images/' + filename), cv2.COLOR_BGR2RGB), (256,256))
    predicted_mask = my_model.predict(original_image.reshape(1,256,256,3))
    predicted_mask_argmax = np.argmax(predicted_mask, axis=3)

    cv2.imwrite("static/images/prediction.jpg", predicted_mask_argmax[0]*20)
