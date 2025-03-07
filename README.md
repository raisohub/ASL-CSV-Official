This GitHub repo was adapted from the following GitHub repository: [https://github.com/mg343/Sign-Language-Detection](url). Any credit to this repository should be directed towards that one.

The model in this repository is also trained using the followign Kaggle dataset: [https://www.kaggle.com/datasets/datamunge/sign-language-mnist](url)

# Introduction #

The following GitHub repository contains the code used by the Responsible AI Student Organization (RAISO) in their Workshop series during the Winter 2025 quarter at Northwestern University. During this workshop series, members were taught through weekly exercises how to build an AI model for the purposes of detecting ASL signs and translating them into their English letters. The project specifically detects 24 of the 26 letters in the English alphabet, as "J" and "Z" both require motion to be expressed in ASL. The resulting code was placed in this GitHub repository.

As a broad overview, this GitHub repository contains the following files:
- `camera.py`, which contains the code to run the program
- `model.py`, which contains the code to train and save the model
- `requirements.txt`, which contains the libraries needed to run the code
- `Winter Coding Exercise.ipynb`, which contains the exercises from the penultimate meeting of the workshop series

The next section of this README will go over how to run the code.

# Setup #

## Prerequisities ##

The following will be required to run the code for this project:
- An editor to edit code (in production VSCode was used)
- Python (preferably the latest version)
- pip to install packages (you can read the instructions here to install it: [https://pip.pypa.io/en/stable/installation/](url))

## Step 1: Pulling the GitHub Repository ##

First, click the green "Code" button on this page. You'll see a couple of options for how to get the files in the GitHub repository onto this device (i.e. locally). For simplicity, we'd recommend downloading the ZIP file and then unzipping its contents. If you prefer another method, however, feel free to do so.

## Step 2: Installing Libraries ##

Go to your terminal and make sure that you are in the folder with the code from the GitHub repository (to make sure, type ls in the terminal, and you should see only the files mentioned above). Afterwards, write `pip install -r requirements.txt` to ensure you have the right libraries for this project. If you have experience with conda or similar tools, feel free to use them.

## Step 3: Running the Code ##

Head over to your editor and run `model.py`. This should create a new file called model.keras, which is the trained model saved onto your device. Next, run `camera.py`. If everything is correct, you should see a new window pop up that uses your camera. Congratulations! You can now use the model to detect ASL.

# Using the Project #

While it is mentioned in `camera.py`, it bears mentioning the controls for the project:

- Hold up your hand with a sign language symbol. If your hand is detected, a green box should surround it
- Press the space bar to analyze the ASL symbol. Your terminal will display the output, written as the top three guesses in alphabetical order followed by their confidence
- To exit the project, press the escape button instead of pressing the "X" in the top right of the application
