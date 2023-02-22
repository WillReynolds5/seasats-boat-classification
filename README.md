<h1>Seasats Autonomous Seacraft Image Classification</h1>
<p>This is an image classification model for the Seasats Autonomous Seacraft, which classifies images as containing a boat or not containing a boat. The model is based on the Vision Transformer (ViT) architecture and is trained on a dataset of labeled images of the sea surface.</p>

<h2>Installation and Project Setup</h2>
<p>To set up the project, first clone the repository:</p>

<code>git clone https://github.com/WillReynolds5/seasats-image-classification.git </code>
<p>Next, create a new conda environment from the provided environment.yml file:</p>

<code>conda env create -f environment.yaml</code>
<p>Activate the environment:</p>

<code>conda activate shipclassifier</code>
<p>You should now be able to run the model and train it on new data.</p>

<h2>Prepare Dataset</h2>
<p>This code crops and resizes images to size=256px, input images can be of any aspect ratio / size</p>
<p>TODO: This should be optimized to what ever aspect ratio and resolution the Seastats camera will have</p>

<p>Images are expected to be in the following directory structure:</p>
<ul>
<li> 70% of the images at /dataset/train, with boat and not boat images in separate folders</li>
<li> 30% of the images at /dataset/val, with boat and not boat images in separate folders</li>
</ul>


<h2>Choosing Hyperparameters</h2>
<p>The ViT model has several hyperparameters that can be tuned to achieve better performance on a particular task. Here are some tips for choosing hyperparameters for the SEASAT dataset:</p>
<ul>
<li>image_size: This parameter controls the size of the input images. In general, larger images may require a larger image_size to capture more details, but may also require more processing power and longer training times. You can experiment with different values to find the best tradeoff between performance and speed.</li>
<li>patch_size: This parameter controls the size of the image patches used by the transformer. Larger patches may capture more contextual information, but may also introduce more noise or reduce the resolution of the input images. Smaller patches may be more precise, but may require more processing power to process. You can experiment with different values to find the best balance.</li>
<li>dim: This parameter controls the dimension of the transformer embeddings. Higher values may allow the model to capture more complex relationships between patches, but may also require more processing power and longer training times. You can experiment with different values to find the best tradeoff between performance and speed.</li>
<li>depth: This parameter controls the number of transformer layers. Deeper models may be able to capture more complex patterns, but may also require more processing power and longer training times. You can experiment with different values to find the best tradeoff between performance and speed.</li>
<li>heads: This parameter controls the number of attention heads in each transformer layer. More heads may allow the model to capture more fine-grained patterns, but may also require more processing power and longer training times. You can experiment with different values to find the best tradeoff between performance and speed.</li>
<li>mlp_dim: This parameter controls the dimension of the multi-layer perceptron used in each transformer layer. Higher values may allow the model to capture more complex patterns, but may also require more processing power and longer training times. You can experiment with different values to find the best tradeoff between performance and speed.</li>

</ul>

<h2>Training</h2>
<p>To train the model on new data, just run the train.py script. But first, choose your training parameters</p>
<p>-- epochs: The number of epochs to train for. (default: 10)</p>
<p>-- batch_size: The batch size for training. (default: 32)</p>
<p>-- lr: The learning rate for the optimizer. (default: 1e-4)</p>

<h3>The model learned what a boat is after 10 epochs!</h3>
![Alt text](/losses.png "Is it a boat?")

<h2>Evaluation</h2>
<p>To evaluate the trained model on a new image, you can use the evaluate.py script. Eval has been built with a command line interface so the project can be interfaced as a subprocess from another programming language (what ever you guys are using for hardware, C++?). The script takes the following arguments:</p>
<ul>
<li>image_path: The path to the image file to evaluate.</li>
<li>--model_path: The path to the saved model file. (default: "checkpoints/model.pth")</li>
</ul>
<p>For example, to evaluate the model on an image file named boat.jpg, you could run:</p>
<code>python evaluate.py boat.jpg</code>
<br>
<p>The script will preprocess the image using the preprocess_data function, run it through the pre-trained model, and output the model's prediction for the image ("boat" or "not boat"). The default path for the model file is model.pt, but you can specify a different path using the --model_path argument:</p>
<code>python evaluate.py boat.jpg --model_path my_model.pth</code>
<br>
<p>TODO: do not randomly crop images for EVAL</p>

<h2>TODO</h2>
<p>make the model work on non square images</p>
<p>Optimize the preprocessing for the seasats dataset</p>
<p>ViT is optimzied to run on GPU, a different model may be more well suited for the seasats hardware (CNN/RESNET)</p>
