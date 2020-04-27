import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

video = '.\\test_video.mp4'
videoLeftUp = cv2.VideoCapture(video)
videoLeftDown = cv2.VideoCapture(video)
videoRightUp = cv2.VideoCapture(video)
videoRightDown = cv2.VideoCapture(video)


fps = videoLeftUp.get(cv2.CAP_PROP_FPS)

width = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_HEIGHT)))


videoWriter = cv2.VideoWriter('.\\4in1.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, (width, height))


while(videoLeftUp.isOpened()):
    successLeftUp, frameLeftUp = videoLeftUp.read()
    successLeftDown , frameLeftDown = videoLeftDown.read()
    successRightUp, frameRightUp = videoRightUp.read()
    successRightDown, frameRightDown = videoRightDown.read()
    if successLeftUp == True:
        
        frameLeftUp = cv2.resize(frameLeftUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        cv2.putText(frameLeftUp, 'Original', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)


        frameLeftDown = cv2.resize(frameLeftDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        tf.compat.v1.enable_eager_execution()
        rgb_2_gray = tf.image.rgb_to_grayscale(frameLeftDown)
        rgb_2_gray_narray = rgb_2_gray.numpy()
        frameLeftDown = cv2.cvtColor(rgb_2_gray_narray, cv2.COLOR_GRAY2BGR)
        cv2.putText(frameLeftDown, 'Gray', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)



        frameRightUp = cv2.resize(frameRightUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size**2
        frameRightUp = cv2.filter2D(frameRightUp, -1, dst=-1, kernel=kernel, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
        cv2.putText(frameRightUp, 'Gaussian Blur', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)


        frameRightDown = cv2.resize(frameRightDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        hsv_image = cv2.cvtColor(frameRightDown, cv2.COLOR_BGR2HSV)
        hsv_image[:,:,2] = cv2.equalizeHist(hsv_image[:,:,2])
        frameRightDown = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        cv2.putText(frameRightDown, 'Histogram equalization', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        
        
        frameUp = np.hstack((frameLeftUp, frameRightUp))
        frameDown = np.hstack((frameLeftDown, frameRightDown))
        frame = np.vstack((frameUp, frameDown))
        
        videoWriter.write(frame)
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

videoLeftUp.release()
videoLeftDown.release()
videoRightUp.release()
videoRightDown.release()
videoWriter.release()
cv2.destroyAllWindows()
