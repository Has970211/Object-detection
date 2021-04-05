# Object Detection

## How to run
  ```
  git clone https://github.com/Has970211/Object-detection.git
  cd Object-detection

  # install project 
  pip install -e .
  pip install -r requirements.txt
  ```
  
## Imports

  ```
  from ObjectDetection_detectron2.Inference.inference import test
  
  #set paths
  output_dir = '....' #set the folder path of output weight folder that we have supplied
  img_folder = '....' #set the folder path of test image folder(which only contains image files, another files or folders are not allowed to exist.)
  output_folder = '....' #set the folder path to save output images and json files. New "Images" folder and "JsonFile" folder will be created in that folder and output images an json file consists with output predicted results for all images will be saved inside those folders respectively. 
  threshold_scr = ....
  
  Test = test(output_dir, img_folder, threshold_scr, output_folder)
  Test.call()
  
  #output will be saved in the output_folder 
  ```

  

