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
>* :surfer: [Training Parameters](#surfer-training-parameters-classification)
>   * [Termination Conditions](#termination-conditions-classification)
>* :space_invader: [Structure of the Neural Network](#space_invader-structure-of-the-neural-network)
>* :chart_with_downwards_trend: [Training Process Progress Chart](#chart_with_downwards_trend-training-process-progress-chart-classification)
>* :1234: [Contingency Matrix (plotconfusion)](#1234-contingency-matrix-plotconfusion)
>* :paw_prints: [Procedure for Testing Selected 5 Points](#paw_prints-procedure-for-testing-selected-5-points)

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
    <td><img src="https://github.com/user-attachments/assets/1b7756b6-89fe-4a46-8088-be122edffc2a" width="200"></td>
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
