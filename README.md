# Neural-Style-Transfer

# Project Overview
In this project, I have used cycleGAN deep learning model in order to transform an image by applying stylistic features of a particular art form onto the image to generate visually appealing results. CycleGAN is capable of transforming images from one domain to another without paired examples. It employs two generator-discriminator pairs to learn the mappings between domains while maintaining cycle consistency to ensure the transformations are valid and reversible.

# Installation Instructions
To view the results generated after training the model, follow the given below steps:

1.Install the python file given in app.py from GitHub, and save the python file with the name app.py on your desktop.

2.Create a folder(say,by the name of project) and inside the folder, import the app.py file which you had just stored on the computer. 

3.Inside the same folder(project folder) create another folder by the name of templates.

4.Now, open VS Code and select File->Open Folder and select the project folder which you have just created.

5.Now, right click on the templates option and select new file, and name the new file index.html.

6.Now, copy the code under index.html in this repo and paste it in index.html. 

7. Now, click on app.py and afterwards download the weights of the 3 models used(their google drive links are given below) and store in on your computer.

8. Copy the paths of these weights and pase it in the respective generator_f.load_weights("") function.

9. Press Ctrl+S and click on run option. A link where the website is hosted will be share on the terminal. Upon clicking that link you will be redirected to the

   website where you can choose the different styles and upload images and get the generated images.

10. If you want to run the code on google colab, you can run the google colab notebook and in the last line of the notebook, you can select how many images to take from the UC Berkeley dataset and click on run option. The results will be generated. But before testing, ensure that you have mounted google drive to your colab notebook, have the model weights inside the google drive and copy the path of those model weights and paste it where the weights are loaded on the notebook(generator_f.load_weights() function).

11. In the google colab notebook, generator_f is for van gogh, generator_f1 is for ukiyoe and generator_f2 is for cezanne




https://drive.google.com/file/d/15wZeZGrkh0lBrzn1T8yvyEn1XpOiGtR3/view?usp=sharing   #Van-Gogh Model

https://drive.google.com/file/d/1-iqLi6z15BZpAiOYfaDmu5AIOAUytzcs/view?usp=sharing   #Ukiyo-e Model

https://drive.google.com/file/d/14Ep7XGvRjOQL56Rf6hrwUvNc4qQ__D5y/view?usp=sharing   # Cezanne Model

# Google Colab Notebook

https://colab.research.google.com/drive/1U_Ur_wK1D_iKTCMtp37-SG97LEXg57DQ?usp=sharing

# Dependencies
tensorflow, tensorflow_addons, matplotlib, os, time, numpy, Flask,io,PIL

