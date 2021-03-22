# ATTIC
AuTomaTic Issue Classifier (ATTIC)

Authors:
* Quentin Perez
* Pierre-Antoine Jean ([Github](https://github.com/PAJEAN) :octocat:)

Co-authors
* Christelle Urtado
* Sylvain Vauttier


# Description
This repository contains source code about experiments about binary bug tickets classifiers.
It is structured in 5 directories :file_folder: .

```bash
root
|
├─ requirements.txt
├─ LICENSE
├─ README.md
├─ genetic_algorithm 
├─ classifier_selection 
├─ multilayer_perceptron_settings
├─ data
```


#### genetic_algorithm 
Directory containing sources implementing genetic algorithm to optimize hyper-parameters (features 
number and hidden_layer_size for the MLP)

#### classifier_selection 
Contains a Jupyter notebook to reproduce results used to compare 6 classifiers (MLP, SVM, SGD, RR, RF, KNN).

####  multilayer_perceptron_settings
Contains a Jupyter notebook implementing 5 different settings with TF-IDF and MLP.

#### data
Json files containing the dataset of 5,591 tickets used in experiments. 
Dataset :scroll: is split into 7 files to avoid the anonymisation limitation of 1MB performed by _anonymous.4open.science_
The dataset coming from the conference paper: 
"[It’s not a bug, it’s a feature: how misclassification impacts bug prediction](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/05/icse2013-bugclassify.pdf)" 
by Herzig, Kim and Just, Sascha and Zeller, Andreas.)

# How to get data and source code ? 

To get data and source code you can use this command in a Linux terminal : 
```bash
$ mkdir paper_96 && \ 
cd paper_96 && \ 
wget -r https://anonymous.4open.science/r/816dccc2-ed04-4c0a-871a-bd517c06fa5a/ && \ 
cd anonymous.4open.science/repository/816dccc2-ed04-4c0a-871a-bd517c06fa5a/ && \
mv * ../../.. && \
cd ../../.. && \
rm -rf anonymous.4open.science
```

# How to use ? :computer:

### Requirements 
* Python 3
* Package Installer for Python (pip)

You can install Python required libraries using the following command:
```bash
$ pip install -r requirements.txt
```


### genetic_algorithm
Genetic algorithm parameters are set in a configuration Python file named "genetic_algo_params.py" at the directory root.

A python main name "genetic_main.py" is runnable with the following command:
```bash
$ python3 genetic_main.py
```

To run it in background with logging:
```bash
$ nohup python3 genetic_main.py > genetic_algo.log &
```

### classifier_selection
To run the Jupyter notebook please run this command:
```bash
$ jupyter notebook baseline_notebook.ipynb
```

###  multilayer_perceptron_settings
To run the Jupyter notebook please run this command:

```bash
$ jupyter notebook mlp_settings.ipynb
```
