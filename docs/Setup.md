# Setup - Quick Guide

- [Info](Info)
- [Style Guide](Style-Guide)
- [VS-Code Extensions](Recommended-VS-Code-Extension)
- [Virtual Environments](Virtual-Environments)


## Info
The `requirements.txt` provided only works for the new api of gymnasium.

## Style Guide

To ensure a consistent codebase, we mostly follow the [PEP 8 Style Guide](https://peps.python.org/pep-0008/) for Python code.  
To enforce part of this, we use the Python linter `flake8` and the formatter `black`.  
We modify `flake8` to be in sync with our formatter. The max-line-length is set to 88 (from 79, this is where we deviate from the PEP 8 Style Guide) and we ignore the Error `E203` and `W503`.  
To do that, just modify (or create) your `.flake8` file the following way:
```ini
[flake8]
ignore = E203
ignore = W503
max-line-length = 88
```
To format on save and to set Black as the default formatter, add the following to your VS Code`settings.json` file:
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
We recommend the built in module venv, but others are also possible (virtualenv, anaconda maba, miniconda, ...).
The following commands are for the module venv and work for Unix systems in the terminal (tbd).
To ensure all the packages in our requirements.txt work, you might need to install the Python development headers. To ensure `python.h` is available, you must install the appropriate development package for Python using your system's package manager (Typically included on MacOS).

#### Fedora

```bash
sudo dnf install python3-devel
```
For python3.10:
```bash
sudo dnf install python3.10-devel
```

#### MacOS

```bash
brew install python
```
For python3.10:
```bash
brew install python@3.10
```

#### Creating virtual environments:

Virtual environments are created by executing the venv module:
```bash
python -m venv /path/to/new/virtual/environment
```

To create a venv with a python version, e.g Python 3.10 (note that the Python version has to be installed first)
```bash
python3.10 -m venv /path/to/new/virtual/environment
```

#### Activate the virtual environment

```bash
source <venv>/bin/activate
```

#### Deactivate:

```bash
deactivate
```

#### Managing Packages:

Install required packages:
```bash
pip install <package>
```
Create requirements.txt:
```bash
pip freeze > requirements.txt
```
Install packages from requirements.txt:
```bash
pip install -r requirements.txt
```
