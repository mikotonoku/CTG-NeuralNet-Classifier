# CTG Neural Network Classifier :pushpin:
The purpose of the task was to create and train a neural network capable of classifying (recognizing) an infant's condition based on **CTG examination** data. The network should use **a maximum of 60% of the data for training and 40% for testing**.

#### :dart: The goal was to:
* Achieve a classification accuracy **greater than 92%** on test data.
* Validate the trained network by running **five repetitions** with randomly split data (in my case).
* Compare **at least three different network structures**.
* Perform **testing** on samples from **each disease category** using the best-trained network.

*Content of the documentation* :arrow_down:
## Content:
>* :repeat: [Description of Input and Output Data](#repeat-description-of-input-and-output-data)
>* :page_facing_up: [MLP Network Structure](#page_facing_up-mlp-network-structure)
>* :surfer: [Training Parameters](#surfer-training-parameters)
>   * [Termination Conditions](#termination-conditions)
>   * [Criterion Function](#criterion-function)
>* [Training Process and Contingency Matrix for the Best Network](#training-process-and-contingency-matrix-for-the-best-network)
>   * :chart_with_downwards_trend: [Training Process Progress Chart](#chart_with_downwards_trend-training-process-progress-chart)
>   * :1234: [Contingency Matrix (plotconfusion)](#1234-contingency-matrix-plotconfusion)
>* :clipboard: [Neural Network Testing](#clipboard-neural-network-testing)
>* :page_facing_up: [Training Process and Contingency Matrix for Different Neuron Counts](#page_facing_up-training-process-and-contingency-matrix-for-different-neuron-counts)
>   * [First Variant: 100 Neurons](#first-variant-100-neurons)
>   * [Second Variant: 10 Neurons](#second-variant-10-neurons)
>* :pill: [Testing Samples from Each Disease Type for the Best-Trained Network](#pill-testing-samples-from-each-disease-type-for-the-best-trained-network)
>* :page_facing_up: [Classification Accuracy, Sensitivity, and Specificity](#page_facing_up-classification-accuracy-sensitivity-and-specificity)

### :repeat: Description of Input and Output Data
#### *INPUT DATA*
*The data comes from the file `CTGdata.mat` and includes **25 parameters** derived from measured signals from **cardiotocography (CTG) examination**.*
>```matlab
>% Loading data from the file
>data = load('CTGdata.mat');
>```
These parameters are stored in the variable `NDATA` and serve as inputs for the neural network.

#### *OUTPUT DATA*
*The variable `typ_ochorenia` contains classification into three groups:*
* **1** = Normal condition
* **2** = Suspect condition
* **3** = Pathological condition
>```matlab
>targets = dummyvar(data.typ_ochorenia);      
>% Creating a binary representation of categorical values
>```
These groups are transformed into **binary representation (one-hot encoding)** using the `dummyvar` function before being used in the neural network.

#### *DATA SPLITTING INTO TRAINING AND TESTING*
*Setting parameters for **random data splitting** into training and testing sets:*
>```matlab
>net.divideFcn = 'dividerand';  
>net.divideParam.trainRatio = 0.6;                % 60% training data
>net.divideParam.valRatio = 0;  
>net.divideParam.testRatio = 0.4;                 % 40% test data
>```

>:arrow_left: [**Back to *CONTENT***](#content)

### :page_facing_up: MLP Network Structure
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/e267da7b-0d18-4fb4-b5d7-6e2838f4093c" width="200"></td>
    <td>
      <b><i>Inputs:</i></b>
      <p>The network receives <b>25 input parameters</b>, stored in the variable <code>NDATA</code>. These parameters are derived from measured signals from <b>cardiotocography (CTG) examination</b>.</p>
      <ul>
        <li><b>Neuron type:</b> Input neurons.</li>
        <li><b>Number of neurons:</b> 25 (same as the number of input parameters in NDATA).</li>
      </ul>
      <b><i>Hidden Layer:</i></b>
      <ul>
        <li><b>Neuron type:</b> Neurons with a <b>nonlinear activation function</b> (in this case, the default function <code>tansig</code>—hyperbolic tangent sigmoid function).</li>
        <li><b>Number of neurons:</b> 32 (set by the variable <code>hidden_neurons</code>).</li>
      </ul>
      <b><i>Output Layer:</i></b>
      <p>The network outputs <b>three disease classification categories</b>:</p>
      <ol>
        <li><b>Normal</b> condition</li>
        <li><b>Suspect</b> condition</li>
        <li><b>Pathological</b> condition</li>
      </ol>
      <ul>
        <li><b>Neuron type:</b> Neurons with a <code>softmax</code> <b>activation function</b> (typically used for classification tasks).</li>
        <li><b>Number of neurons:</b> 3 (one for each disease category).</li>
      </ul>
    </td>
  </tr>
</table>

>:arrow_left: [**Back to *CONTENT***](#content)

### :surfer: Training Parameters
```matlab
net.trainParam.goal = 0.001; % Termination condition for error
net.trainParam.epochs = 1000; % Maximum number of epochs
net.trainParam.max_fail = 12; % Maximum number of failed validations

% Training the neural network and returning training data
[net, tr] = train(net, data.NDATA', targets');
```
>:arrow_left: [**Back to *CONTENT***](#content)

#### *Termination Conditions:*
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/03ae3fa0-c4d3-4529-87a4-a7938e9a2709" width="200"></td>
    <td>
      <ul>
        <li><i><code>net.trainParam.goal = 0.001;</code></i></li>
          <ul>
            <li>Termination condition for error: Training stops if the network error reaches <b>0.001</b>.</li>
          </ul>
        <li><i><code>net.trainParam.epochs = 1000;</code></i></li>
          <ul>
            <li>Maximum number of epochs: The network can go through a maximum of <b>1000 training cycles</b>.</li>
          </ul>
        <li><i><code>net.trainParam.max_fail = 12;</code></i></li>
          <ul>
            <li>Maximum number of validation failures: Training stops if, after <b>12 consecutive failed validations</b>, no improvement occurs.</li>
          </ul>
      </ul>
    </td>
  </tr>
</table>

>:arrow_left: [**Back to *CONTENT***](#content)

#### *Criterion Function:*
The criterion function is **Mean Squared Error (MSE)**. This function is used to evaluate the network's error during training.

In the code:
* Training stops when the MSE reaches the set **target value** of 0.001 (`net.trainParam.goal = 0.001`).

>:arrow_left: [**Back to *CONTENT***](#content)

### Training Process and Contingency Matrix for the Best Network
> :grey_exclamation: *The best network was obtained with 32 initial neurons.*

#### :chart_with_downwards_trend: *Training Process Progress Chart:*
![image](https://github.com/user-attachments/assets/5bc66d02-3694-4d74-855c-3b325d56911f)

The chart illustrates the **training process** of the neural network and the **error reduction** over training epochs.

#### Key Elements of the Chart:
:large_blue_circle: *Blue Line (Training Data Error):*
   * Shows the error trend on training data across epochs.
   * Initially, the error **decreases sharply**, indicating that the model is learning.
   * As training progresses, the error **stabilizes**, suggesting convergence.

:red_circle: *Red Line (Testing Data Error):*
   * Represents the error on unseen test data.
   * A **stable test error** suggests **strong generalization** to new data.
   * If the test error **increases**, it may indicate **overfitting**, where the model performs well on training data but struggles with new data.

:paperclip: *Best Performance Point:*
   * Marks the epoch where the model achieved its **lowest error value**, demonstrating peak accuracy during training.
    
:paperclip: *Overall Trend:*
   * A gradual **decrease in error** signifies that the model is successfully adjusting its parameters.
   * If the gap between **training and test error** is too large, it may indicate **poor generalization**.

>:arrow_left: [**Back to *CONTENT***](#content)

#### :1234: *Contingency Matrix (plotconfusion):*
![image](https://github.com/user-attachments/assets/23b96b36-6f42-4110-9442-79481fd3053a)

🔷 *Purpose of the Contingency Matrix:*
* Provides a **detailed evaluation** of the model’s classification accuracy.
* Helps **identify misclassified instances** and improve model performance.

📊 *Interpreting the Matrix:*
* **Diagonal Values (Green):** Represent correctly classified samples, indicating strong classification accuracy.
* **Off-Diagonal Values (Red):** Show misclassified instances; fewer red values suggest better generalization.

🚀 *Performance Metrics:*
* **Training Accuracy:** 100%
* **Testing Accuracy:** 93.3%
* **Error Rate:** 6.7%

📎 *Significance of Results:*
* High **training accuracy** confirms effective learning of patterns.
* Stable **testing accuracy** ensures proper **generalization**, preventing overfitting.

![image](https://github.com/user-attachments/assets/cdaefd01-b96d-416f-bdc6-7635df70f62a)

🔷 *Overall Classification Accuracy:*
* This contingency matrix provides a **comprehensive assessment** of the neural network’s performance, considering both training and testing results.
* The classification **accuracy** across all data is **97.3%**, meeting the expected requirements.

📊 *Evaluation Summary:*
* **Training Accuracy:** 100%
* **Testing Accuracy:** 93.3%
* **Overall Accuracy:** 97.3%

📎*Key Insights:*
* The high classification accuracy indicates that the model effectively **learns patterns** and generalizes well.
* A minimal **error percentage** suggests strong reliability and robustness in predictions.

>:arrow_left: [**Back to *CONTENT***](#content)

### :clipboard: *Neural Network Testing:*
>🔎 **Testing Method:** The model was evaluated using the **"5-time training with random data splitting"** approach. This ensures that the network’s classification performance is **consistent and robust** across multiple runs.

> 
📊 **Average Results Across 5 Runs:**
* Training Accuracy: min = 99.9%, **average = 99.9%**, max = 99.9%
* Testing Accuracy: min = 92.56%, **average = 92.56%**, max = 92.56%

📎 **Performance Evaluation:**
* The model successfully **meets the requirement** of exceeding **92% classification accuracy** on test data.
* The **minimal variation** across different runs confirms that the model maintains **stability and reliability** in classification.

📊 **MATLAB Implementation:**
```matlab
test_accuracies = 1 - confusion(test_target, test_outputs);                                 % Calculate success rate  
fprintf('Training Accuracy: min = %.2f%%, avg = %.2f%%, max = %.2f%%\\n', ...  
min(train_accuracies) * 100, mean(train_accuracies) * 100, max(train_accuracies) * 100);  
fprintf('Testing Accuracy: min = %.2f%%, avg = %.2f%%, max = %.2f%%\\n', ...  
min(test_accuracies) * 100, mean(test_accuracies) * 100, max(test_accuracies) * 100);  
```

>:arrow_left: [**Back to *CONTENT***](#content)

### :page_facing_up: Training Process and Contingency Matrix for Different Neuron Counts
#### First Variant: 100 Neurons
The number of neurons was set to **100** as an illustrative example. 
> :grey_exclamation: Typically, changing the number of neurons results in ***minor deviations** from previously documented successful outcomes.
![image](https://github.com/user-attachments/assets/8d3645ba-c3e1-4ac8-80e0-d7d1b5dd6e2f)

#### Key Elements of the Chart:
:large_blue_circle: *Blue Line (Training Data Error):*
   * At the start of training, the **error value** on training data dropped **only during the first epoch** to **2.627**.
   * The curve then **remained constant** throughout the **1000 epochs**, indicating **suboptimal learning progression**.
   * This suggests that the model **stagnated**, failing to effectively optimize weights in the hidden layer.

:red_circle: *Red Line (Testing Data Error):*
   * The **test error remained at the same level** as the training error, **indicating poor generalization**.
   * **No improvement** was observed in test accuracy, suggesting the model failed to **capture the underlying structure** of the test data.

📊 **Contingency Matrix Analysis (100 Neurons):**
![image](https://github.com/user-attachments/assets/a4a6a2f0-5173-47ce-9583-322fd50a821d)

> * Training Accuracy: **87.7%**, Testing Accuracy: **89.8%**, Error Rate: **10.2%**

![image](https://github.com/user-attachments/assets/58839f21-b805-4e1a-a97c-cba2f2d250b5)

> * The contingency matrix reveals that the **overall classification accuracy** is **88.5%**, which is **below the acceptable threshold**.

📎 **Key Insights:**
* The network **did not learn effectively**, resulting in **high error rates**.
* The model **failed to optimize** correctly, leading to poor performance on unseen data.
* **Increasing the neuron count** beyond an optimal number **does not necessarily improve classification accuracy**.

>:arrow_left: [**Back to *CONTENT***](#content)

#### Second Variant: 10 Neurons
**Neuron Configuration:** For the second variant, the number of **hidden neurons was set to 10** to evaluate its impact on performance.
![image](https://github.com/user-attachments/assets/f00ab20b-bf23-4db5-97b9-0f1a1fa23206)

#### Key Elements of the Chart:
:large_blue_circle: *Blue Line (Training Data Error):*
   * The **error stabilized** after a few epochs but remained **higher** than when using 32 neurons.
   * The model **learned slowly** and failed to **optimize weights effectively**, indicating that 10 neurons were **insufficient** for capturing data complexity.

:red_circle: *Red Line (Testing Data Error):*
   * **Higher test error** compared to training error suggests **weaker generalization**.
   * Though the test error stabilized, it remained **significantly worse** than results with 32 neurons.

📊 **Contingency Matrix (10 Neurons):**
![image](https://github.com/user-attachments/assets/2b50f9df-25c5-4799-8bd4-cc5b53913fc2)

📎 **Key Insights:**
> * The matrix offers a **detailed evaluation** of classification accuracy for training and testing.
> * Training Accuracy: **98.7%**, Testing Accuracy: **90.2%**, Error Rate: **9.8%** → *Insufficient for optimal performance*

![image](https://github.com/user-attachments/assets/a88e7381-429d-4655-9ecb-7ab780034dba)

📎 **Overall Classification Performance:**
* The matrix considers both training and testing results.
* Total Classification Accuracy: **95.3%** → *Acceptable but still below desired levels*

🚀 **Final Observations:**
* The **low neuron count** resulted in **limited learning capacity**, leading to **high error rates**.
* While classification accuracy improved over training, the network **struggled with unseen data**, limiting generalization.

>:arrow_left: [**Back to *CONTENT***](#content)

### :pill: Testing Samples from Each Disease Type for the Best-Trained Network
![image](https://github.com/user-attachments/assets/b414ac48-5039-40ce-b235-d14e0d113fc7)
> :grey_exclamation: The result on the screenshot is **in Slovak**, while the output in the published code is **in English**.

🔎 **Overview:**  
The image displays the **classification results** for samples from each disease category using the **best-trained neural network** (with an initial neuron count of 32).  

📊 **Sample Classification Details:**  

1️⃣ **Normal Sample**  
- **Predicted Probabilities:** **[1, 2.9031e-21, 5.3626e-29]**  
  - The **first value (1)** indicates nearly **100% certainty** that the sample belongs to Group 1 (*Normal*).  
  - The remaining probabilities are **close to zero**, showing that the model is highly confident in this classification.  

2️⃣ **Suspicious Sample**  
- **Predicted Probabilities:** **[5.0451e-14, 0.99996, 3.7645e-05]**  
  - The **second value (0.99996)** indicates a **very high probability** that the sample belongs to Group 2 (*Suspicious*).  
  - The first and third probabilities are **nearly zero**, suggesting the model is strongly confident in this classification.  

3️⃣ **Pathological Sample**  
- **Predicted Probabilities:** **[1.3813e-16, 3.2275e-12, 1]**  
  - The **third value (1)** demonstrates almost **100% certainty** that the sample belongs to Group 3 (*Pathological*).  
  - The other probability values are **negligible**, reinforcing the model’s confidence in its decision.  

📎 **Key Insights:**  
- The **high probability values** for the correct classifications indicate that the neural network is **highly reliable** in distinguishing between different disease types.  
- The **low probability values** for incorrect classifications suggest the model’s **certainty and precision** in predictions.  

>:arrow_left: [**Back to *CONTENT***](#content)

### :page_facing_up: Classification Accuracy, Sensitivity, and Specificity

📊 **Overview:**  
The image presents the **evaluation metrics** for classification accuracy, sensitivity, and specificity.  

🔵 **Sensitivity:**  
- **99.32% on the training set** → The model correctly identified **almost all positive cases**.  
- **85.88% on the test set** → Slightly lower accuracy on unseen data, but still **a strong performance**.  

🔴 **Specificity:**  
- **99.69% on the training set** → The model accurately recognizes **normal cases**.  
- **95.54% on the test set** → Still **high precision** in identifying negative cases within test data.  

📎 **Overall Accuracy:**  
- **99.22% on the training set** → The model achieves **near-perfect accuracy** on training data.  
- **91.88% on the test set** → Accuracy declined but remains **above the standard success threshold**.

>:arrow_left: [**Back to *CONTENT***](#content)
