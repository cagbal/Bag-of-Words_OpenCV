
This program calculates the Harris and SIFT Features of two different images and then applies Bag-of-Words methods on training images and
calculates the histogram of visual words by using OpenCV API. 



Please put two different images for finding Harris/SIFT features and all 20 training images and 4 test images for BOW on same directory
with exe. 
You need to name training images as "train_" + training_image_number(train_1, train_2...), also the test images like "test_1", "test_2" ..
and the image format must be jpeg.

Run Command: 

name_of_program.exe living_room.jpg living_room_2.jpg 

then program will help user to navigate.

Before compiling, please add lib_color and lib_features libraries to project.

Credits

- Images from SUN database
- OpenCV for Image Processing and Vision Methods