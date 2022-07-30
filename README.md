# AutoEncoders-for-Flowers

In this project I applied various architectures of traditional autoencoders on various flowers species to study, analyze and generate new ones.

The project is object oriented, I have made an Autoencoder class to use and reuse it for different hierarchies, I also have added a customized loss function to control the variation between neighbouring pixels in the generated images and force continuity, furthermore I built an image generator class from scratch that flow images directly from the images directory and take care of compatibility and formatting whitout having to store into memory which is prohibitively expensive, additionally, I built a callback class that track an image during the training and visualizes the resulting output in real time using openCV, in order to judge the performence of the current model.

<br/><br/> 

- the first model was trained only on rose flowers, an encoder with six Convolutional layers and likewise for the decoder with a latent space of dimension 32
after having been trained the model was tested on some images, 8 images got encoded and then decoded to give the following results:

the tracked image:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/tracked_images/rose_gif.gif?raw=true)

True vs Predicted:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/figures/rose_true_vs_predicted.png?raw=true)

we took the last two images in the latter figure, encoded them and then added them up (average) in the latent space then decoded them back to the original space, this is the result:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/figures/rose_blendnig_last_tow.png?raw=true)

<br/><br/> 

- the second model consists of a five layers encoder, we are trying to strangle the input -sunflowers- and downtransform it to a two-dimensional vector, another five layers decoder is used to decode it back to the original space, it's worth noting that the goal of recovering the original image is unattainable, because we can't project a whole image rich of informations into the plane and recover it enirely without loss of information, those are some results:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/figures/sunflower_true_vs_predicted.png?raw=true)

we took the first and third sunflowers projected them into the plane, we then selected uniformely 400 points residing in the rectangle covered by those two images,
after having decoded those points, here are the results:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/figures/sunflower_morphing_first_and_third.png?raw=true)

<br/><br/> 

- the third model was trained on both iris and calendula flowers with a latent space size of 16, this is more challenging than the first model, in which both image size is of (128, 128), here is the result:

the tracked image:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/tracked_images/iris_gif.gif?raw=true)

True vs Predicted:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/figures/iris_and_calendula_true_vs_predicted.png?raw=true)

we took the third image -an iris- and the fifth one -a calendula-, encoded them, combined them, then decoded the resulting vector, here it is:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/figures/third_and_fifth_iris_calendula.png?raw=true)

here we used different values of weights in the weighted average, an alpha varying from 0.1 to 0.9, the results are as follows:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/figures/different_average_weights_iris_calendula.png?raw=true)

<br/><br/> 

- the fourth model was trained on the entire flower dataset , here are the results:

the tracked image:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/tracked_images/flower_gif.gif?raw=true)

True vs Predicted:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/figures/flowers_true_vs_predicted.png?raw=true)

we also made a histogram plot for each dimension of the latent space after having projected all the dataset into that space, here are the histograms:

![alt text](https://github.com/TheMagicShop/AutoEncoders-for-Flowers/blob/main/figures/flowers_latent_space_distribution.png?raw=true)




<br/><br/> 
<br/><br/> 
<br/><br/> 
<br/><br/> 

the data is from kaggle, Licensed under CC0: Public Domain, we directly download it using this script:

```
# kaggle.json is downloadable token from kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d l3llff/flowers
```
