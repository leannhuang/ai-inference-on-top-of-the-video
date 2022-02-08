from ast import arguments
import cv2
import os
import numpy as np
import collections
import sys
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

def main():
    # define your object here 
    #object_list = ['Filet-O-Fish', 'Soda', 'French Fries', 'McNuggets']
    object_set = ('hamburger', 'soda', 'fries', 'mcnuggets')

    # arguments
    argument_list = sys.argv
    print(argument_list)
    if len(argument_list) < 2:
        print('You did not assgin the probabily in the command argument. Set the probabilty to default value 50')
        prob_threshold = 0.5   
    else:
        prob_threshold = argument_list[1]
    
    # custom vision credentials information
    prediction_key = '<Your Prediction Key>'
    ENDPOINT = '<Your Prediction ENDPOINT>'
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(endpoint=ENDPOINT, credentials=credentials)
    project_id = '<Your Custom Vision Project Id>'
    PUBLISH_ITERATION_NAME = 'Iteration1'
    
    # video path information
    video_path = './'
    video_name = '<Your Video Name>'
    parent_path = './'
    extract_img_folder = project_id + '_extracted_images'

    frame_count = extract_frames_from_video(video_path, video_name, parent_path, extract_img_folder)
    ai_inference(frame_count, extract_img_folder, predictor, project_id, PUBLISH_ITERATION_NAME, prob_threshold, object_set, parent_path)
    compose_video(frame_count, PUBLISH_ITERATION_NAME, prob_threshold, project_id)


def extract_frames_from_video(video_path, video_name, parent_path, extract_img_folder):
    vidcap = cv2.VideoCapture(video_path + video_name)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    if success == 'False':
        print('1. Read the video FAILED. Check if your video file exists')
    else:
        print(f'1. Read the video SUCESSFULLY. The fps of the video is {fps}')
  
    img_path = parent_path + extract_img_folder
    os.mkdir(img_path)
    frame_count = 0
    
    while success:
        
        cv2.imwrite(os.path.join(img_path , f'frame_{frame_count}.jpg'), image)     
        success,image = vidcap.read()
        frame_count += 1
    
    print('2. Finish extracting the video to frames')
    return frame_count

def generate_color(object_set):
    object_list = list(object_set)
    object_count = len(object_list)
    object_to_color = collections.defaultdict()

    for i in range(object_count):
        color = np.random.choice(range(256), size=3)
        color = (int(color[0]), int(color[1]), int(color[2]))
        object = object_list[i]
        object_to_color[object] = color

    print('4. The colors of the objects are fully assigned')
    return object_to_color

def ai_inference(frame_count, extract_img_folder, predictor, project_id, PUBLISH_ITERATION_NAME, prob_threshold, object_set, parent_path):

    # Open the sample image and get back the prediction results.
    with open(os.path.join(extract_img_folder, "frame_0.jpg"), mode="rb") as test_data:
        results = predictor.detect_image(project_id, PUBLISH_ITERATION_NAME, test_data)
        print(results)
        print('3. Call the Custom Vision SUCESSFULLY')
    
    prob = float(prob_threshold)
    
    tagged_folder = f'{project_id}_tagged_images'
    path = os.path.join(parent_path, tagged_folder)
    os.mkdir(path)

    # generate colors of the bbox for diffent objects
    object_to_color = generate_color(object_set)

    for i in range(frame_count):
        
        with open(os.path.join(extract_img_folder, "frame_%d.jpg" % i), mode="rb") as test_data:
            results = predictor.detect_image(project_id, PUBLISH_ITERATION_NAME, test_data)
            img = cv2.imread(f'{extract_img_folder}/frame_%d.jpg' % i)
            shape = img.shape
            filtered_preds = [prediction for prediction in results.predictions if prediction.probability > prob and prediction.tag_name in object_set] 

            for pred in filtered_preds:

                x = int(pred.bounding_box.left * shape[1])
                y = int(pred.bounding_box.top * shape[0])
                
                start_point = (x, y)
                
                x2 = x + int(pred.bounding_box.width * shape[1])
                y2 = y + int(pred.bounding_box.height * shape[0])
                
                end_point = (x2, y2)
                
                img = cv2.rectangle(img, start_point, end_point, object_to_color[pred.tag_name], 5)
                img = cv2.putText(img, pred.tag_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, object_to_color[pred.tag_name], 5)                 

            cv2.imwrite(f"{tagged_folder}/tagged{i:04d}.jpg", img)
    
    print('5. Finish inferencing the frames of the video')

def compose_video(frame_count, PUBLISH_ITERATION_NAME, prob_threshold, project_id):
    tagged_folder = f'{project_id}_tagged_images'
    video_name = f'tagged_{PUBLISH_ITERATION_NAME}_{prob_threshold}.avi'
    img = cv2.imread(f'{project_id}_tagged_images/tagged0000.jpg')
    shape = img.shape

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out_video = cv2.VideoWriter(video_name, fourcc, 25, (shape[1], shape[0]))

    for i in range(frame_count):
        file_name = f'{tagged_folder}/tagged{i:04d}.jpg'
        img = cv2.imread(file_name)
        out_video.write(img)
        
    out_video.release()

    print('6. Finish composing the video')
    print(f'7. Check the inferenced video - {video_name} under the ai-inerence-on-top-of-video folder')


if __name__ == "__main__":
    main()

