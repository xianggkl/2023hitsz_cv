# Assignment 1

## How to start working on an assignment?

Before working on each assignment, you need to setup a few things:

1. **Install Python 3.6+:** 

To use python3, make sure to install **version 3.6+** on your local machine. (For convenience, you don't need to create virtual environment)

2. **Change pip mirror:**

Run the following commands to change mirror of pip, then you can install python packages faster.

```sh
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip 
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

3. **Install necessary packages:** 

To install necessary packages, run the command **in your assignment directory** :

```sh
pip install -r requirements.txt
```

4. **Working with [Jupyter notebooks](https://jupyter.org/) :** 

**In your assignment directory**, start the notebook with the `jupyter notebook` command. 

When working with a Jupyter notebook, you can edit the `*.py` files either in the Jupyter interface (in your browser) or with your favorite editor (vim, Atom, vscode...). Whenever you save a `*.py` file, the notebook will reload their content directly.
