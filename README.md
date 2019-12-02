# Konduit Serving examples

This repository contains examples for using [Konduit Serving](https://serving.oss.konduit.ai).

## Run examples

To run the examples in this repository, you will have to install a supported version of the Java Development Toolkit (JDK) and Python. JDK version 8 and Python 3.7+ is recommended. 

Install [Git](https://git-scm.com/) and run the following command in your terminal:

```
git clone https://github.com/KonduitAI/konduit-serving-examples.git
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the konduit package. 

```bash
pip install konduit
# if you have git and python3 installed
konduit-python --os windows-x86_64
```

You may have to install PyTorch or TensorFlow to run some of the examples in this repository.  

Detailed installation steps are available [here](https://serving.oss.konduit.ai/installation). 


## Examples

### Python
1. Serving a model built with TensorFlow 1.x
    - [01_tensorflow-basic-mnist.ipynb](notebooks/01_tensorflow-basic-mnist.ipynb)
    - [02_tensorflow-basic-bert.ipynb](notebooks/02_tensorflow-basic-bert.ipynb)
2. Serving a model built with Deeplearning4j: [03_deeplearning4j-multilayernetwork.ipynb](notebooks/03_deeplearning4j-multilayernetwork.ipynb)
2. Serving an ONNX model file with ONNX Runtime (PyTorch, CNTK, MXNet)
   - PyTorch: [05_onnx_pytorch.ipynb](notebooks/05_onnx_pytorch.ipynb)

### Java 
1. Configuring a DataVec transform process: [BasicConfiguration.java](java/src/main/java/ai/konduit/serving/examples/basic/BasicConfiguration.java)
2. Loading and transforming an image file: [BasicConfigurationImage.java](java/src/main/java/ai/konduit/serving/examples/basic/BasicConfigurationImage.java)
2. Serving a Python script: [BasicConfigurationPython.java](java/src/main/java/ai/konduit/serving/examples/basic/BasicConfigurationPython.java)
