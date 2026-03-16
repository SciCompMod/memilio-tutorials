# MEmilio Tutorials

Check out the main code repository at https://github.com/SciCompMod/memilio .

The official documentation can be found here https://memilio.readthedocs.io .

### Marimo notebook setup

First, open a new terminal.
Go to a directory for the new project, then create a virtual environment (here "venv") and install marimo:

``` sh
python -m venv venv
source venv/bin/activate
python -m pip install marimo
```

Note that on Windows, you need to use `source .\venv\Scripts\activate` instead.

The notebook can then be started by running `marimo edit` (within the venv we just created and activated).
This should open a new browser window with the notebook. Otherwise, look for and open the link with `http://localhost:2718` from marimo's output.

The notebook can *later* be closed through the red X in the top right of the marimo browser tab, or by pressing `Ctrl+C` in the terminal twice.

You should see the tutorials in the `Workspace` section (within the marimo browser tab). Otherwise, try re-running `marimo edit` from the directory containing this Readme.

Tutorials can be started by clicking a file in the Workspace section. To switch from an opened file to another, you can use the `Files` view on the top left, or return to the homepage via the hamburger menu on the top right and selecting `Return home`.

### Quick install

To execute the tutorials, we require the memilio-simulation package as well as further dependencies as listed in `requirements.txt`. You can install them in a new virtual environment as follows where we clone the MEmilio repository into this directory to install the memilio-simulation package:
```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

git clone git@github.com:SciCompMod/memilio.git
cd memilio/pycode/memilio-simulation
pip install . 

```
