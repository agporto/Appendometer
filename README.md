# Appendometer

This is the source code for the machine learning based version of the drosophila `Appendometer`. It performs landmark prediction in drosophila images using the [ml-morph](https://github.com/agporto/simple-ml-morph) pipeline.

<p align="center"><img src="https://github.com/agporto/Appendometer/blob/master/resources/logo2.png" width="800"></p>


## Install

1. Clone the repo:
```
git clone https://github.com/agporto/Appendometer && cd Appendometer/
```

2. Create a clean virtual environment 
```
conda create -n append python=3.7
conda activate append
```
3. Install dependencies
````
python -m pip install --upgrade pip
pip install -r requirements.txt
````

## Usage

Once the packages are installed, you can use the codebase by running the `appendML.py` command line interface. This file takes in several **optional** arguments, which are described below:

* `-i` or `--input-dir`: (optional) The input directory (default = images)
* `-p` or `--predictor`: (optional) The trained shape prediction model (default = resources/predictor.dat)
* `-o` or `--out-file`: (optional) The output filename suffix (default = output.xml)
* `-l` or `--ignore-list`: (optional) prevents landmarks of choice from being output
* `-m` or `--max-error`: (optional) maximum prediction error in pixels

Example prompt:

```
python appendML.py -i images/ -p resources/predictor.dat -m 42
```

## Citation

```
Coming soon!
```
