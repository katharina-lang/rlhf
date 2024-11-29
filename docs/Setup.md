# Setup - Quick Guide

- [Style Guide](Style-Guide)
- [VS-Code Extensions](Recommended-VS-Code-Extension)
- [Virtual Environments](Virtual-Environments)


## Style Guide

To ensure a consistent codebase, we mostly follow the [PEP 8 Style Guide](https://peps.python.org/pep-0008/) for Python code.  
To enforce part of this, we use the Python linter `flake8` and the formatter `black`.  
We modify `flake8` to be in sync with our formatter. The max-line-length is set to 88 (from 79, this is where we deviate from the PEP 8 Style Guide) and we ignore the Error `E203`.  
To do that, just modify (or create) your `.flake8` file the following way:
```ini
[flake8]
ignore = E203
max-line-length = 88
```
To format on save and set Black as the default formatter, update your VS Code`settings.json` file:
```json
"[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
}
```

## Recommended VS-Code Extension

- Python
- Black Formatter
- Flake8
- Jupyter
- Pylance
- Tensorboard
- vscode-icons
- vscode-pdf


## Virtual Environments

#### What is a Virtual Environment?

A virtual environment is a tool to create isolated Python environments for projects, keeping dependencies separate.
W recommend the built in module venv, but others are also possible (virtualenv, anaconda maba, miniconda, ...).
The following commands are for the module venv and work for Unix systems in the terminal (tbd).

#### Creating virtual environments:

Virtual environments are created by executing the venv module:

`python -m venv /path/to/new/virtual/environment`

To create a venv with a python version, e.g Python 3.10 (note that the Python version has to be installed first)

`python3.10 -m venv /path/to/new/virtual/environment`

#### Activate the virtual environment

`source <venv>/bin/activate</code>`

#### Deactivate:

`deactivate`

#### Managing Packages:

Install required packages:

`pip install <package>`

Create requirements.txt:

`pip freeze > requirements.txt`

Install packages from requirements.txt:

`pip install -r requirements.txt`
