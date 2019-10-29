import os
import sys

def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))


#src_path = os.path.join(get_parent_dir(1),'2_Training','src')
utils_path = os.path.join(get_current_dir(),'Utils')
src_path = os.path.join(get_current_dir(),'src')

sys.path.append(src_path)
sys.path.append(utils_path)

import argparse
from keras_yolo3.yolo import YOLO
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# Set up folder names for default values
data_folder = os.path.join(get_current_dir(),'Data')

image_folder = os.path.join(data_folder,'Source_Images')

image_test_folder = os.path.join(image_folder,'Test_Images')

detection_results_folder = os.path.join(image_folder,'Test_Image_Detection_Results') 
detection_results_file = os.path.join(detection_results_folder, 'Detection_Results.csv')

model_folder =  os.path.join(data_folder,'Model_Weights')

model_weights = os.path.join(model_folder,'trained_weights_final.h5')
model_classes = os.path.join(model_folder,'data_classes.txt')

anchors_path = os.path.join(src_path,'keras_yolo3','model_data','yolo_anchors.txt')

FLAGS = None


class BoxBounds:
    def __init__(self,label,xmin,ymin,xmax,ymax):
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

def GetInfoFromPath(filepath):
    dirname = os.path.dirname(filepath)
    completefilename = os.path.basename(filepath)
    info = os.path.splitext(completefilename)
    filenamenoextension = info[0]
    extension = info[1]

    return dirname,filenamenoextension,extension

def CropBox(InputImagePath,OutputImagePath,xmin,ymin,xmax,ymax):

    img = Image.open(InputImagePath)
    area = (xmin, ymin, xmax, ymax)
    cropped_img = img.crop(area)
    cropped_img.save(OutputImagePath)


def detectUpperAndLowerBoundsPrediction(predictions,upperClasses,lowerClasses,max_height):

    ## Classes: LIKES, BODY, COMMENTS, RETWEETS, DATE, HANDLE
    UpperBoundFound = False # there could be no Date, or handle picked up
    LowerBoundFound = False # there could be no comment, retweet or likes picked up
    UpperBound = 0 # represents the Upper Bound
    LowerBound = max_height # Represents the lower Bound
 
    for p in predictions:
        # If prediction is on an upper object
        if(p[4] in upperClasses):

            # xmin, ymin, xmax, ymax, label, confidence level
            # We need to maximum of all y_max
            if(p[3] > UpperBound):
                UpperBoundFound = True 
                UpperBound = p[3]


        if(p[4] in lowerClasses):

            if(p[3] < LowerBound):
                LowerBoundFound = True
                LowerBound = p[1]

    return UpperBoundFound,LowerBoundFound, UpperBound,LowerBound

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        "--input_images", type=str, default=image_test_folder,
        help = "Path to image directory. All subdirectories will be included. Default is " + image_test_folder
    )

    parser.add_argument(
        "--output", type=str, default=detection_results_folder,
        help = "Output path for detection results. Default is " + detection_results_folder
    )

    parser.add_argument(
        "--no_save_img", default=True, action="store_true",
        help = "Only save bounding box coordinates but do not save output images with annotated boxes. Default is False."
    )

    parser.add_argument(
        '--yolo_model', type=str, dest='model_path', default = model_weights,
        help='Path to pre-trained weight files. Default is ' + model_weights
    )

    parser.add_argument(
        '--anchors', type=str, dest='anchors_path', default = anchors_path,
        help='Path to YOLO anchors. Default is '+ anchors_path
    )

    parser.add_argument(
        '--classes', type=str, dest='classes_path', default = model_classes,
        help='Path to YOLO class specifications. Default is ' + model_classes
    )

    parser.add_argument(
        '--confidence', type=float, dest = 'score', default = 0.25,
        help='Threshold for YOLO object confidence score to show predictions. Default is 0.25.'
    )

    csvout = os.path.join(detection_results_folder,"_GLOBAL_Results.csv")

    parser.add_argument(
        '--result_file', type=str, dest = 'result_file', default = csvout,
        help='Path to Global CSV Result File'
    )

    parser.add_argument(
        '--box_file', type=str, dest = 'box', default = detection_results_file,
        help='File to save bounding box results to. Default is ' + detection_results_file
    )

    FLAGS = parser.parse_args()

    #FLAGS.model_path = model_weights
    #FLAGS.anchors_path = anchors_path
    #FLAGS.classes_path = model_classes
    #FLAGS.score = 0.25
    #FLAGS.box = detection_results_file
    FLAGS.postfix = '_withBox'
    FLAGS.gpu_num = 1
    #FLAGS.no_save_img = False
    #FLAGS.output = detection_results_folder
    #FLAGS.input_images = image_test_folder

    result_file = FLAGS.result_file

    save_img = not FLAGS.no_save_img

    input_image_paths = GetFileList(FLAGS.input_images)

    #print('Found {} input images: {}...'.format(len(input_image_paths), [ os.path.basename(f) for f in input_image_paths[:5]]))

    output_path = FLAGS.output
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # define YOLO detector
    yolo = YOLO(**{"model_path": FLAGS.model_path,
                "anchors_path": FLAGS.anchors_path,
                "classes_path": FLAGS.classes_path,
                "score" : FLAGS.score,
                "gpu_num" : FLAGS.gpu_num,
                "model_image_size" : (416, 416),
                }
               )

    # Make a dataframe for the prediction outputs
    #out_df = pd.DataFrame(columns=['image', 'image_path','xmin', 'ymin', 'xmax', 'ymax', 'label','confidence','x_size','y_size'])

    # labels to draw on images
    class_file = open(FLAGS.classes_path, 'r')
    input_labels = [line.rstrip('\n') for line in class_file.readlines()]
    #print('Found {} input labels: {}...'.format(len(input_labels), input_labels))

    start = timer()
    text_out = ''

    global_out_df = pd.DataFrame(columns=['filename','image_path','label','xmin', 'ymin', 'xmax', 'ymax','width','height'])

    for i, img_path in enumerate(input_image_paths):
        #print(img_path)

        out_df = pd.DataFrame(columns=['image_path','label','xmin', 'ymin', 'xmax', 'ymax'])

        prediction, image = detect_object(yolo, img_path, save_img = save_img,
                                          save_img_path = FLAGS.output,
                                          postfix=FLAGS.postfix)
        y_size,x_size,_ = np.array(image).shape


        HBFound, LBFound, YMIN_BODY, YMAX_BODY = detectUpperAndLowerBoundsPrediction(prediction,[4,5],[0,2,3],y_size)

        AllBoxes = []

        for single_prediction in prediction:
            #print(single_prediction)
            # xmin, ymin, xmax, ymax, label, confidence level
            #[646, 24, 732, 48, 4, 0.35650748]
            ## Classes: LIKES, BODY, COMMENTS, RETWEETS, DATE, HANDLE
            category = single_prediction[4]
            confidence = single_prediction[5]
            if(category == 1): # = if it is BODY..
                if (HBFound  and LBFound):

                    xmin = 0 #single_prediction[0]-60
                    ymin = YMIN_BODY
                    xmax = x_size #single_prediction[2]+30
                    ymax = YMAX_BODY

                    xmin_upper = 0
                    ymin_upper = 0
                    xmax_upper = x_size
                    ymax_upper = ymin+20


                if( not HBFound and LBFound):

                    xmin = 0 #single_prediction[0]-60
                    ymin = 0
                    xmax = x_size #single_prediction[2]+30
                    ymax = YMAX_BODY

                    xmin_upper = 0
                    ymin_upper = 0
                    xmax_upper = x_size
                    ymax_upper = 80

                if(HBFound and not LBFound):

                    xmin = 0 #single_prediction[0]-60
                    ymin = YMIN_BODY
                    xmax = x_size #single_prediction[2]+30
                    ymax = y_size

                    xmin_upper = 0
                    ymin_upper = 0
                    xmax_upper = x_size
                    ymax_upper = ymin+20

                if( not HBFound and not LBFound):

                    xmin = 0 #single_prediction[0]-60
                    ymin = 0
                    xmax = x_size #single_prediction[2]+30
                    ymax = y_size

                    xmin_upper = 0
                    ymin_upper = 0
                    xmax_upper = x_size
                    ymax_upper = 80

            else:

                xmin = single_prediction[0]-20
                ymin = single_prediction[1]-5
                xmax = single_prediction[2]+30
                ymax = single_prediction[3]+10


            oneBox = BoxBounds(input_labels[category],xmin,ymin,xmax,ymax)
            AllBoxes.append(oneBox)

            if(category == 1):

                AllBoxes.append(BoxBounds("UPPER",xmin_upper,ymin_upper,xmax_upper,ymax_upper))


        print("\n")
        print("File: "+img_path)
        print("LABEL,X_MIN,Y_MIN,X_MAX,Y_MAX\n")
        for b in AllBoxes:
            if(b.label != "HANDLE"):
                #print('{} : MIN:[{},{}] MAX:[{},{}]'.format(b.label,b.xmin,b.ymin,b.xmax,b.ymax) )
                #'image_path','label','xmin', 'ymin', 'xmax', 'ymax'


                print('{},{},{},{},{}'.format(b.label,b.xmin,b.ymin,b.xmax,b.ymax))
                dirname,namenoExt,extension = GetInfoFromPath(img_path)
                filename = namenoExt+extension
                #df2 = pd.DataFrame(img_path,b.label,b.xmin,b.ymin,b.xmax,b.ymax,columns=['image_path','label','xmin', 'ymin', 'xmax', 'ymax'])

                #out_df=out_df.append(df2)
                new_row = {'image_path':img_path,'label':b.label,'xmin':b.xmin,'ymin':b.ymin,'xmax':b.xmax,'ymax':b.ymax}
                out_df = out_df.append(new_row, ignore_index=True)

                height = b.xmax - b.xmin
                width = b.ymax - b.ymin
                global_new_row = {'filename':filename,'image_path':img_path,'label':b.label,'xmin':b.xmin,'ymin':b.ymin,'xmax':b.xmax,'ymax':b.ymax,'height':height,'width':width}
                global_out_df = global_out_df.append(global_new_row, ignore_index=True)
                


                OutputImagePath = os.path.join(detection_results_folder,namenoExt+"_"+b.label+extension)
                
                #CropBox(img_path,OutputImagePath,b.xmin,b.ymin,b.xmax,b.ymax)

        #CSVOutput = os.path.join(detection_results_folder,namenoExt+"_Results.csv")
        #out_df.to_csv(CSVOutput,index=False)

            #out_df=out_df.append(pd.DataFrame([[os.path.basename(img_path.rstrip('\n')),img_path.rstrip('\n')]+single_prediction + [x_size,y_size]],columns=['image','image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label','confidence','x_size','y_size']))
    end = timer()
    
    global_out_df.to_csv(result_file,index=False)

    #print('Processed {} images in {:.1f}sec - {:.1f}FPS'.format(
    #     len(input_image_paths), end-start, len(input_image_paths)/(end-start)
    #     ))
    #out_df.to_csv(FLAGS.box,index=False)






