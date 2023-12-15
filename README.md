# ml-scratch-pca
Principle Component Analysis Algorithm

## **Description**
The following is my from scratch implementation of the Principal Component Analysis algorithm.

### **Dataset**

I tested the performance of my model on three datasets: \
\
    &emsp;1. Breast Cancer Dataset \
    &emsp;2. Iris Dataset \
    &emsp;3. Diabetes Dataset

### **Walkthrough**

**1.** Need the following packages installed: sklearn, numpy, and matplotlib.

**2.** Once you made sure all these libraries are installed, evrything is simple, just head to main.py and execute it.

**3.** Since code is modular, main.py can easily: \
\
    &emsp;**i.** Load the three datasets \
    &emsp;**ii.** Transform the features to chosen number of components \
    &emsp;**iii.** Plot scatter plots of transformed data.

### **Results**

For each dataset I will share the train feature shapes before and after transformation.

**1.** Breast Cancer Dataset:

- Before Transformation:
     - Shape = (569, 30)
- After Transformation:
     - Shape = (569, 2)

- See visualization below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-pca/blob/main/plots/bc/bc_pca.png?raw=true)

**2.** Iris Dataset:

- Before Transformation:
     - Shape = (150, 4)
- After Transformation:
     - Shape = (150, 2)

- See visualization below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-pca/blob/main/plots/iris/iris_pca.png?raw=true)

**2.** Diabetes Dataset:

- Before Transformation:
     - Shape = (442, 10)
- After Transformation:
     - Shape = (442, 2)

- See visualization below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-pca/blob/main/plots/db/db_pca.png?raw=true)