# Object Detection

## How to run
  ```
  git clone https://github.com/Has970211/Object-detection.git
  cd Object-detection

  # install project 
  pip install -e .
  pip install -r requirements.txt
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  ```
  
## Imports

  ```
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

  

