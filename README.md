# Alive-Or-Dead-Human-Detection
This project implements a deep learning approach in Python using TensorFlow to detect and classify human activity (alive/dead) in video footage captured by a camera.



# Main Model architecture

The inputs of the main model are 4 images of a humman in different time-frames (at least 30ms difference between each frame) with size of (None , 256 ,256 , 3) and the output would be 1(alive) or 0(dead).

![bg right:44%](https://github.com/SAhmadrezaAnaami/Alive-Or-Dead-Human-Detection/blob/main/images/mainModel.png)


As you can see our main model Consist of Base Mdoel and a secondary Model. Base model is aan Inception Model in witch weights are freezed and a secondary model.


# Inception Model

![bg right:44%](https://buddhism.net/wp-content/uploads/2024/02/6a6522d8-cb48-13e9-af77-d3d96ec3eddd.jpg)
link to image : https://buddhism.net/wp-content/uploads/2024/02/6a6522d8-cb48-13e9-af77-d3d96ec3eddd.jpg
