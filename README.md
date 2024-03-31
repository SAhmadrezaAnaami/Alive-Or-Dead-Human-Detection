# Alive-Or-Dead-Human-Detection
This project implements a deep learning approach in Python using TensorFlow to detect and classify human activity (alive/dead) in video footage captured by a camera.



# Main Model architecture

The inputs of the main model are 4 images of a humman in different time-frames (at least 30ms difference between each frame) with size of (None , 256 ,256 , 3) and the output would be 1(alive) or 0(dead).

![bg right:44%](https://github.com/SAhmadrezaAnaami/Alive-Or-Dead-Human-Detection/blob/main/images/mainModel.png)
