    ##################################################################
    #                                                                #
    #           SADECE ADAM GIBI ADAMLAR 2000 SATIR KODU             #
    #                                                                #
    #                   TEK DOSYAYA YAZARLAR                         #
    #                                                                #
    #                                                                #
    #                 --Mustafa Kemal Atatürk--                      #
    #                                                                #
    ##################################################################

import threading
import numpy as np
import cv2
import time
import argparse
import sys
import ogl_viewer.tracking_viewer as gl
import pyzed.sl as sl
from multiprocessing import Process
import math



def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("resolution" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")

parser = argparse.ArgumentParser()
parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
parser.add_argument('--imu_only', action = 'store_true', help = 'Either the tracking should be done with imu data only (that will remove translation estimation)' )
opt = parser.parse_args()




if (len(opt.input_svo_file)>0 and len(opt.ip_address)>0):
    print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
    exit()




init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                             coordinate_units=sl.UNIT.METER,
                             coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
parse_args(init_params)


zed = sl.Camera()
image_size = zed.get_camera_information().camera_configuration.resolution




image_size.width = 1280/2
image_size.height = 720/2
image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
init_params.camera_fps=30
point_cloud = sl.Mat()


#camera = image_zed.get_data()
runtime_parameters = sl.RuntimeParameters()


status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open", status, "Exit program.")
    exit(1)

tracking_params = sl.PositionalTrackingParameters() #set parameters for Positional Tracking
tracking_params.enable_imu_fusion = True
status = zed.enable_positional_tracking(tracking_params) #enable Positional Tracking
if status != sl.ERROR_CODE.SUCCESS:
    print("Enable Positional Tracking : "+repr(status)+". Exit program.")
    zed.close()
    exit()

runtime = sl.RuntimeParameters()
camera_pose = sl.Pose()

camera_info = zed.get_camera_information()
def tetete():
    # Preparing variables for spatial dimensions of the frames
    h, w = None, None

    # Loading COCO class labels from file
    # Opening file
    # Pay attention! If you're using Windows, yours path might looks like:
    # r'yolo-coco-data\coco.names'
    # or:
    # 'yolo-coco-data\\coco.names'
    with open('detector/obj.names') as f:
        # Getting labels reading every line
        # and putting them into the list
        labels = [line.strip() for line in f]

    # loading config file and weights
    network = cv2.dnn.readNetFromDarknet('detector/yolov4-tiny-custom.cfg',
                                         'detector/yolov4-tiny-custom_last.weights')

    # Getting list with names of all layers from YOLO v3 network
    layers_names_all = network.getLayerNames()

    # Getting only output layers' names that we need from YOLO v3 algorithm
    # with function that returns indexes of layers with unconnected outputs
    layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

    # Setting minimum probability to eliminate weak predictions
    probability_minimum = 0.5

    # Setting threshold for filtering weak bounding boxes
    # with non-maximum suppression
    threshold = 0.3

    # Generating colours for representing every detected object
    # with function randint(low, high=None, size=None, dtype='l')
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    # milimetre cinsinden değerleri girelim.

    focalLength = 200
    tabela_uzunlugu = 500
    bilinen_uzaklik = 784

    # def lidarCallback(x):
    #     print("lidar",x.lidar)

    # rospy.Subscriber("/lidar", Lidar, lidarCallback)

    def focal_lenght_calculator(bilinen_uzaklik, box_height, tabela_uzunlugu):
        global focalLength
        focalLength = (bilinen_uzaklik * box_height) / tabela_uzunlugu

    def distance_to_camera(tabela_uzunlugu, focalLength, pixel_uzunlugu):
        return (tabela_uzunlugu * focalLength) / pixel_uzunlugu
        # Create OpenGL viewer
    if opt.imu_only:
        sensors_data = sl.SensorsData()
    py_translation = sl.Translation()
    pose_data = sl.Transform()

    text_translation = ""
    text_rotation = ""
    file = open('output_trajectory.csv', 'w')
    file.write('tx, ty, tz \n')



    # Defining loop for catching frames
    while True:
        # Capturing frame-by-frame from camera
        #
        # _, frame = camera.read()
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)

            zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH)

            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        tracking_state = zed.get_position(camera_pose,
                                          sl.REFERENCE_FRAME.WORLD)  # Get the position of the camera in a fixed reference frame (the World Frame)
        if opt.imu_only:
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                rotation = sensors_data.get_imu_data().get_pose().get_euler_angles()
                text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
        else:
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                # Get rotation and translation and displays it
                rotation = camera_pose.get_rotation_vector()
                translation = camera_pose.get_translation(py_translation)
                print("------------------------------------------------")
                print(f"tX: {translation.get()[0]}\ntY: {translation.get()[1]}\ntZ: {translation.get()[2]}")
                text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2),
                                        round(translation.get()[2], 2)))
                pose_data = camera_pose.pose_data(sl.Transform())
                file.write(str(translation.get()[0]) + ", " + str(translation.get()[1]) + ", " + str(
                    translation.get()[2]) + "\n")
            # Update rotation, translation and tracking state values in the OpenGL window

        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)



            image_ocv = image_zed.get_data()
            depth_image_ocv = depth_image_zed.get_data()
            # frame = image_ocv[:, :, :-1]
            # frameN = np.ascontiguousarray(frame, dtype=np.uint8)
            frame = np.ascontiguousarray(image_ocv[:, :, :-1], dtype=np.uint8)
            ##################################
            # Getting spatial dimensions of the frame
            # we do it only once from the very beginning
            # all other frames have the same dimension
            if w is None or h is None:
                # Slicing from tuple only first two elements
                h, w = frame.shape[:2]

            # Getting blob from current frame
            # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob from current
            # frame after mean subtraction, normalizing, and RB channels swapping
            # Resulted shape has number of frames, number of channels, width and height
            # E.G.:
            # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)

            # Implementing forward pass with our blob and only through output layers
            # Calculating at the same time, needed time for forward pass
            network.setInput(blob)  # setting blob as input to the network
            start = time.time()
            output_from_network = network.forward(layers_names_output)
            end = time.time()

            # Showing spent time for single current frame
            # print('Current frame took {:.5f} seconds'.format(end - start))

            # Preparing lists for detected bounding boxes,
            # obtained confidences and class's number
            bounding_boxes = []
            confidences = []
            class_numbers = []

            # Going through all output layers after feed forward pass
            for result in output_from_network:
                # Going through all detections from current output layer
                for detected_objects in result:
                    # Getting 80 classes' probabilities for current detected object
                    scores = detected_objects[5:]
                    # Getting index of the class with the maximum value of probability
                    class_current = np.argmax(scores)
                    # Getting value of probability for defined class
                    confidence_current = scores[class_current]

                    # # Check point
                    # # Every 'detected_objects' numpy array has first 4 numbers with
                    # # bounding box coordinates and rest 80 with probabilities
                    # # for every class
                    # print(detected_objects.shape)  # (85,)

                    # Eliminating weak predictions with minimum probability
                    if confidence_current > probability_minimum:
                        # Scaling bounding box coordinates to the initial frame size
                        # YOLO data format keeps coordinates for center of bounding box
                        # and its current width and height
                        # That is why we can just multiply them elementwise
                        # to the width and height
                        # of the original frame and in this way get coordinates for center
                        # of bounding box, its width and height for original frame
                        box_current = detected_objects[0:4] * np.array([w, h, w, h])

                        # Now, from YOLO data format, we can get top left corner coordinates
                        # that are x_min and y_min
                        x_center, y_center, box_width, box_height = box_current
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))

                        # Adding results into prepared lists
                        bounding_boxes.append([x_min, y_min,
                                               int(box_width), int(box_height)])
                        confidences.append(float(confidence_current))
                        class_numbers.append(class_current)

            # Implementing non-maximum suppression of given bounding boxes
            # With this technique we exclude some of bounding boxes if their
            # corresponding confidences are low or there is another
            # bounding box for this region with higher confidence

            # It is needed to make sure that data type of the boxes is 'int'
            # and data type of the confidences is 'float'
            # https://github.com/opencv/opencv/issues/12789
            results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                       probability_minimum, threshold)

            # Checking if there is at least one detected object
            # after non-maximum suppression
            if len(results) > 0:
                # Going through indexes of results
                for i in results.flatten():
                    # Getting current bounding box coordinates,
                    # its width and height
                    x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                    box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                    # Preparing colour for current bounding box
                    # and converting from numpy array to list
                    colour_box_current = colours[class_numbers[i]].tolist()

                    # # # Check point
                    # print(type(colour_box_current))  # <class 'list'>
                    # print(colour_box_current)  # [172 , 10, 127]

                    # Drawing bounding box on the original current frame
                  

                    err, point_cloud_value = point_cloud.get_value(x_center, y_center)
                    if math.isfinite(point_cloud_value[2]):
                        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                             point_cloud_value[1] * point_cloud_value[1] +
                                             point_cloud_value[2] * point_cloud_value[2])
                        print(f"Distance to Camera at ({x_center},{y_center}): {distance}")
                        print("----------------------------------------------------")
                        print(f"coordinates of the object: X:{point_cloud_value[0]}, Y:{point_cloud_value[1]}, "
                              f"Z:{point_cloud_value[2]}")
                    else:
                        print(f"The distance cannot be computed at ({x_center},{y_center})")


                    cv2.rectangle(frame, (int(x_min), int(y_min)),
                                  (int(x_min) + box_width, int(y_min) + box_height),
                                  colour_box_current, 2)

                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                           confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(frame, text_box_current, (int(x_min), int(y_min) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

                    if focalLength == 0:
                        focal_lenght_calculator(bilinen_uzaklik, box_height, tabela_uzunlugu)

                    j = distance_to_camera(tabela_uzunlugu, focalLength, box_height)
                    k = int(j / 10)

                    # b = str(class_numbers[0]) +','+str(x_min)+','+str(y_min)+','+str(box_width)+','+str(box_height)+','+str(k)
                    b = str(class_numbers[0]) + ',' + str(k)

                    print(labels[int(class_numbers[i])], confidences[i])

                    ###Deneme

                    # '1'+','
                    # c = '2'+','+str(class_numbers[0]) +','+str(x_min)+','+str(y_min)+','+str(box_width)+','+str(box_height)

                    # c.data = c
            # Showing results obtained from camera in Real Time11,274,200,67,67,772.2985074626865

            # Showing current frame with detected objects
            # Giving name to the window with current frame
            # And specifying that window is resizable
            # cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
            # Pay attention! 'cv2.imshow' takes images in BGR format

            cv2.imshow('YOLO v3 Real Time Detections', frame)
            # Breaking the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Releasing camera

        # Destroying all opened OpenCV windows


def tracker():

    # Create OpenGL viewer
    if opt.imu_only:
        sensors_data = sl.SensorsData()
    py_translation = sl.Translation()
    pose_data = sl.Transform()

    text_translation = ""
    text_rotation = ""
    file = open('output_trajectory.csv', 'w')
    file.write('tx, ty, tz \n')
    while True:

        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            tracking_state = zed.get_position(camera_pose,sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
            if opt.imu_only :
                if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                    rotation = sensors_data.get_imu_data().get_pose().get_euler_angles()
                    text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
            else :
                if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    #Get rotation and translation and displays it
                    rotation = camera_pose.get_rotation_vector()
                    translation = camera_pose.get_translation(py_translation)
                    print("------------------------------------------------")
                    print(f"tX: {translation.get()[0]}\ntY: {translation.get()[1]}\ntZ: {translation.get()[2]}")
                    text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                    text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
                    pose_data = camera_pose.pose_data(sl.Transform())
                    file.write(str(translation.get()[0])+", "+str(translation.get()[1])+", "+str(translation.get()[2])+"\n")
                # Update rotation, translation and tracking state values in the OpenGL window
        else :
            time.sleep(0.001)

#proctetete = Process(target=tetete)
#proctrack = Process(target=tracker,args=())

thread1 = threading.Thread(target=tracker)

thread2 = threading.Thread(target=tetete)

if __name__ == "__main__":

    thread2.start()
    #thread1.start()

    #proctetete.start()
   # proctrack.start()

