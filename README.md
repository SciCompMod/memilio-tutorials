# MEmilio Tutorials

Check out the main code repository at https://github.com/SciCompMod/memilio .

The official documentation can be found here https://memilio.readthedocs.io .

### Jupyter notebook setup for C++

Make sure you have a fairly recent installation of python and gcc.
On a Linux system, `which gcc python` should print two paths.

First, open a new terminal.
Go to a directory for the new project, then create a virtual environment (here "venv") and install jupyter and the C++ kernel.

``` sh
python -m venv venv
source venv/bin/activate
python -m pip install jupyter jupyter-cpp-kernel
```

The notebook can then be started by running `jupyter-notebook` (within the venv we just created and activated).
This should open a new browser window with the notebook. Otherwise, look for and open the link starting with `http://localhost:8888` from jupyter's output.

The notebook can *later* be closed through the jupyter browser tab `File > Shut Down`, or by pressing `Ctrl+C` in the terminal twice.

To create a notebook using the C++ kernel, select the `Files` tab (within the jupyter browser tab), then select `New > C++ 20` on the right.
This will create a new notebook. Existing notebooks can be directly opened from the `Files` tab and should automatically launch the correct kernel.
