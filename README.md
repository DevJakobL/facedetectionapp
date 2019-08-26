# facedetectionapp


## Instalation
1. Install the requirements.txt.
2. download the neuronal networks for face detection from [Google Drive](https://drive.google.com/open?id=1Y0DnhJiIRuHLa-S3qd1jhoO0_wfLE6Az).
## Usage 
```
usage: app.py [-h] --image IMAGE --frozen FROZEN [--output OUTPUT] [--tb TB]
                [--overview OVERVIEW]
  
  A tool to detect faces in images and cut out the detected faces.
  
  optional arguments:
    -h, --help           show this help message and exit
    --image IMAGE        Set path to the image.
    --frozen FROZEN      Set path to the frozen graph.bp.
    --output OUTPUT      Set path to the detected faces. By default the path is
                         output.
    --tb TB              Must true if you use graphs from Tensorbox. By default
                         it is false.
    --overview OVERVIEW  False if you save only the faces. By default it is
                         False.
```