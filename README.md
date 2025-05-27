# GenderBiasDNN
A working Deep Neural Network that tests gender bias on a student performance dataset.
<br/>
For easier use, it is recommended to use the [google colab link](https://drive.google.com/drive/folders/1CzogI1xWzvYGWzEMDRacy7jWxNia5ZeB?usp=drive_link).
<br/>
If not, see below on useful tips to work around with the python file.
# DNN Model Configuration
You must delete or comment out the hyperparameters you are not using, as well as the Data Preprocessing section you are not using. </br>
You may also configure the hyperparameters however you like. <br/>
| Hyperparameter                             | Description                                                                                                                                                                                                                                              |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ModelName**  | A label for the model used for reference and logging.                                                                                                                                                                                                    |
| **DNN.InputFeatures**                 | The number of input features fed into the model. This should match the number of independent (non-target) variables used from the dataset.                                                                                                               |
| **DNN.LayerNeurons**    | A list representing the number of neurons in each layer of the DNN.  |
| **DNN.Classes**                       | Number of output classes. For regression tasks, this is 1 (predicting a single continuous value like exam score). For classification, this would be higher.                                                                                              |
| **DNN.DropoutRate**                  | Dropout rate applied after layers to prevent overfitting. A rate of `0.1` means 10% of neurons are randomly dropped during each training step.                                                                                                           |
| **Training.MaxEpoch**           | Maximum number of epochs (full passes through the training data) for model training. Higher values allow more learning time but may risk overfitting.                                                                                                    |
| **Training.BatchSize**              | Number of training samples processed in one forward/backward pass. A batch size of 256 balances memory use and training speed.                                                                                                                           |
| **Training.LearningRate**       | The learning rate controls how much the model updates its weights at each step. A rate of 0.01 is fairly typical for small to medium-scale datasets.                                                                                                     |


