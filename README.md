# Physical-Adversarial-Attacks-on-Deep-Learning-based-ISP-Pipelines

## About
We developed a patch-based adverarial attack algorithm to explicitly attack the PyNET-CA image signal processing (ISP) model. The algorithm is able to generate patches of various size and intensity and by default places the patch at the center of the image, although you can easily manipuilate a few variables to control the positioning. It then outputs these images into a seperate folder in Google Drive, you can then download these attacked images and pass them to the PyNET-CA model to be processed.

## The code in the notebook
We seperated our code into various components, there is the MAIN block and then there are EXTRA blocks which are highly useful. We break them down as follows:
	1. MAIN block
 		- Full dataset image patch attack, with SSIM. THIS CODE GENERATES A FGSM PATCH ATTACK, takes in a patch_size, epsilon value, and steps (iterations)
			you can also alter the location of the patch by simply changing the values of center_h and center_w, this determines where the
   			center of the patch is located with respect to the RAW input image of size 448x448
      2. EXTRA CODE BLOCK 1
      		- Single image patch attack, with SSIM. THIS CODE GENERATES A FGSM PATCH ATTACK, takes in a patch_size, epsilon value, and steps (iterations)
			This DOES incorporate SSIM into the loss function

      3. EXTRA CODE BLOCK 2
      		- Single full image attack, with SSIM. THIS CODE GENERATES A FGSM FULL IMAGE ATTACK, full image means the entire image gets perturbed not just a specific portion/patch.This DOES 				incorporate SSIM into the loss function
      4. EXTRA CODE BLOCK 3 
      		- Single image random patch attack, no SSIM. THIS CODE GENERATES A RANDOMIZED PATCH of a given size (patch_size) and at the center of the input image. This does not incorporate SSIM 				into the loss function

      5. EXTRA CODE BLOCK 4 
      		- Full dataset single image random patch attack, no SSIM incorporated as loss (USES MSE instead)

      6. Remainder of code is stuff we used for experimentation and background knowledge, this includes work with RGB images and FGSM attacks.

## These are the steps to get PyNET-CA environment set up and running
1. Create new conda environment
	$ conda create -n pynet python=3.6
2. Activate the new environment
	$ conda activate pynet
3. Install dependencies
	$ conda install pytorch
	$ pip install torchvision
	$ pip install pytorch-mssim
	$ pip install IQA-pytorch
	$ pip install tqdm
4. Cd into repo directory 
	$ cd /skyb-aim2020-public/
5. Make changes to code
	- I am running this without GPU so in main.py we made the following changes
		device = ‘cpu’
		for the two “model.load_state_dict(torch.load()) we added the map_location
			the full line is now: model.load_state_dict(torch.load(os.path.join(args.target_dir, ‘model’, ‘/pretrained_model/result/pynetca/model/model_early_perceptual.pth’ if args.perceptual else ‘model_early_fidelity.pth’), map_location=torch.device(‘cpu’)))

6. Run the script
	$ python mian.py —skip_train —test_dir /Desktop/Adversarial_ISP/skyb-aim2020-public/data/test/huwaei_raw

## Dataset
We subsetted the original ZRR Dataset (https://people.ee.ethz.ch/~ihnatova/pynet.html#dataset) which was in this directory '/test/huawei_raw/', this originally had 1204 items.
Our initial subset we looked for objects we though could be identified by YOLO, this narrowed it down to 107 items, then we ran our YOLO model and found that only 58 were identified 
correctly, according to ground truth photos.
So from here we compiled the final 58 images to utilize for our project; this represents our final dataset size, since each dataset is a pair - the ground truth to the pynetca-enhanced/processed photos.
All images from each dataset are paired and have matching indices. These are the indices of the photos we selected for our final dataset:
indices = {6, 28, 41, 42, 43, 60, 81, 114, 115, 116, 117, 142, 158, 178, 179, 183, 184, 192, 201, 211, 212, 213, 222, 249,
250, 253, 302, 347, 348, 453, 466, 477, 481, 483, 500, 570, 583, 584, 609, 665, 666, 695, 714, 718, 719, 760, 784, 1013, 1046, 1051, 1052, 1064, 1065, 1114, 1115, 1124, 1174, 1189}



