# AutoEncoders-for-Flowers

In this project I applied various architectures of traditional autoencoders on various flowers species to study, analyze and generate new ones.
The project is object oriented, I have made an Autoencoder class to use and reuse it for different hierarchies, I also have added a customized loss function to control the variation between neighbouring pixels in the generated images and force continuity, furthermore I built an image generator class from scratch that flow images directly from the images directory and take care of compatibility and formatting whitout having to store into memory which is prohibitively expensive, additionally, I built a callback class that track an image during the training and visualizes the resulting output in real time using openCV, in order to judge the performence of the current model.

- the first model was trained only on rose flowers, an encoder with six Convolutional layers and likewise for the decoder with a latent space of dimension 32
after having been trained the model was tested on some images, 8 images got encoded and then decoded to give the following results:

[]
 
we took the last two images, encoded them and then added them up (average) in the latent space then decoded them back to the original space, this is the result:

[]

- the second model consists of a five layers encoder, we are trying to strangle the input -sunflowers- and downtransform it to a two-dimensional vector, another five layers decoder is used to decode it back to the original space, it's worth noting that the goal of recovering the original image is unattainable, because we can't project a whole image rich of informations into the plane and recover it enirely without loss of information, those are some results:

[]

we took the first and third sunflowers projected them into the plane, we then selected uniformely 400 points residing in the rectangle covered by those two images,
after having decoded those points, here are the results:

[]

- the third model was trained on both iris and calendula flowers with a latent space size of 16, this is more challenging than the first model, in which both image size is of (128, 128), there are some True-vs-Predicted samples:

[]

we took the third image -an iris- and the fifth one -a calendula-, encoded them, combined them, then decoded the resulting vector, here it is:

[]

here we used different values of weights in the weighted average, an alpha varying from 0.1 to 0.9, the results are as follows:

[]
