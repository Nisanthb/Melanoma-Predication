# Melanoma-Predication
<ins> ### INTRODUCTION </ins> 

Melanoma, a form of skin cancer, necessitates accurate early diagnosis for effective treatment.
This project focuses on the binary classification of melanoma lesions, differentiating between
benign and malignant cases using Efficient-Net architecture. ResNet-50, known
for its deep layers and residual learning, and EfficientNet, which optimizes model efficiency, are
trained and evaluated on a dataset sourced from Kaggle.

This project endeavours to develop and evaluate Deep Learning models tailored to classify melanoma lesions using the Efficient-Net architecture within the PyTorch framework. Specifically, it addresses binary classification (benign vs. malignant) and multiclass classification (nine classes), aiming to achieve high accuracy in identifying skin lesions. Accurate classification is crucial for timely and effective medical interventions, potentially improving patient outcomes.

EfficientNet is a family of convolutional neural networks developed by Google AI that balances
network depth, width, and resolution to achieve high performance. EfficientNet models use a
compound scaling method that uniformly scales all dimensions of depth, width, and resolution
using a simple yet highly effective principle. The EfficientNet-B0 variant was used for this
project due to its efficiency and accuracy.

The dataset was sourced from Kaggle for both Binary and Multiclass classification.

<ins> ### BINARY CLASSIFICATION </ins> 

The images are labelled as either 'benign' or 'malignant', representing the binary nature of the
classification task.

- Training set: 10,691 images
- Validation set: 1,188 images
- Test set: 2,000 images

The architecture used is EfficientNet-B0, and the model was trained using mini-batches of size 32, balancing memory constraints and computational efficiency, as larger batch sizes offer more accurate gradient estimates but require more resources. 
The training process spanned approximately 20 epochs, allowing the model to optimize its parameters and learn complex data patterns. 
A dropout rate of 0.5 was applied, randomly dropping half of the neurons during each iteration to mitigate overfitting by forcing the model to rely on different subsets of neurons. 
Stochastic Gradient Descent (SGD) was the optimizer, updating weights iteratively based on mini-batch gradients. 
The learning rate was set to 0.003 and was possibly adjusted during training to enhance convergence stability. 
Binary Cross-Entropy Loss penalised the divergence between predicted and actual class probabilities, aligning well with the task's requirements.

### OBSERVATIONS

During the training process, the accuracy and loss for both training and validation sets were
monitored. The following plots illustrate the model's performance over the training epochs:

- Training and Validation Accuracy Plot: This plot shows how the accuracy improved
over time, indicating the learning progress of the model.
- Training and Validation Loss Plot: This plot shows the decrease in loss over time,
demonstrating how well the model fits the data.
- The model achieved a high accuracy of 91.45% on the test dataset, indicating its effectiveness in
distinguishing between benign and malignant melanoma lesions.

<ins>  ### MULTICLASS CLASSIFICATION  </ins> 
The dataset for this multiclass classification task was sourced from Kaggle.

- Training set: 20,265 images
- Validation set: 5,066 images
- Test set: 10,000 images

The classes include:

- Melano
- Melanocytic nevus
- Basal cell carcinoma
- Actinic keratosis
- Benign keratosi
- Dermatofibroma
- Vascular lesion
- Squamous cell carcinoma
- Unknown

The architecture used is EfficientNet-B0, optimized with the Adam optimizer, which has a learning rate of 1e-4, making it particularly effective in managing sparse gradients and adapting learning rates during training. The model employs categorical cross-entropy as the loss function, which is ideal for multiclass classification tasks. 
The training process spanned approximately 50 epochs, allowing the model to optimize its parameters and learn complex data patterns. 
To ensure efficient learning, a ReduceLROnPlateau scheduler was implemented to lower the learning rate when the validation loss plateaued, facilitating better model convergence. Additionally, early stopping was used, halting the training if the validation loss did not improve for 5 consecutive epochs, thereby preventing overfitting and conserving computational resources.

### OBSERVATION

During the training process, the accuracy and loss for both training and validation sets were
monitored. The following plots illustrate the model's performance over the training epochs:

- Training and Validation Accuracy Plot: This plot shows how the accuracy improved
over time, indicating the learning progress of the model.
- Training and Validation Loss Plot: This plot shows the decrease in loss over time,
demonstrating how well the model fits the data.
- The model achieved a high accuracy of 86.13% on the test dataset, indicating its effectiveness in
distinguishing between benign and malignant melanoma lesions.

### CONCLUSION

Efficient-Net for Binary and Multi-class classification of melanoma lesions has proven successful, achieving high accuracy and robust performance metrics. This model can significantly aid in the early detection and diagnosis of melanoma, potentially improving patient outcomes.

### FUTURE TARGETS

The future version of this model aims to predict images across any dataset and classify images even in the presence of noise, incorporating robust defence mechanisms to enhance its resilience and accuracy.
