# Corrosion-Detector

## How to run
  ```
  git clone https://github.com/Has970211/Object_Detection.git
  cd corrosion-detector

  # install project 
  pip install -e .
  pip install -r requirements.txt
  ```
  
## Imports

  ```
  python
  from ObjectDetection_detectron2.Inference.inference import test
  
  #set paths
    OUTPUT_DIR = '....'
    data_dir = '....'
    json_dir = '....'
    img_dir = '....'
  
  Test = test(OUTPUT_DIR, data_dir, json_dir, img_dir)
  Test.call()
  
  #output will be saved in the image folder 
  ```

  

