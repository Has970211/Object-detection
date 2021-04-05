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
  output_dir = '....'
  img_folder = '....'
  output_folder = '....'
  
  threshold_scr=....
  
  Test = test(output_dir, img_folder, threshold_scr, output_folder)
  Test.call()
  
  #output will be saved in the image folder 
  ```

  

