# ai-inference-on-top-of-the-video
The goal of this sample code is to guide you to output a video with bbox on top of it by leveraging Azure Custom Vision

input: video

output: inference result on top of the video

## Prerequisites
- Azure Subscription : ([Free trial account](https://azure.microsoft.com/en-us/free/))

## Steps
1. Create an object detection model as in [here](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/get-started-build-detector)
2. Reference the object you defined in custom vision and modify the objects set accordingly[here](https://github.com/leannhuang/ai-inference-on-top-of-the-video/blob/main/ai_inference_bbox.py#L13) 
3. After publishing the iteration, fill in the values of `prediction_key`, `ENDPOINT`,` project_id`,`PUBLISH_ITERATION_NAME` in the custom vision credentials information block[here](https://github.com/leannhuang/ai-inference-on-top-of-the-video/blob/main/ai_inference_bbox.py#L24)
4. Put the video for inference under the ai-inference-on-top-of-the-video folder, and fill in the value of the  `video_name`
5. Open your terminal and type the command with the probablity threshold(first argument) like below 
```
   python ai_inference_bbx_bk.py 0.99
```
6. You will see the log below to know the current process
   1. Read the video SUCESSFULLY. The fps of the video is XX.XXXXX
   2. Finish extracting the video to frames 
   3. Call the Custom Vision api SUCESSFULLY
   4. The colors of the objects bbox are fully assigned
   5. Finish inferencing the frames of the video
   6. Finish composing the video
   7. Check the inferenced video under the ai-inerence-on-top-of-video folder

![commands arguments](docs/images/commands_argument.png)