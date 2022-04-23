# About
Form-NN is a  trained to .
To find out more, check out the provided research paper:
  * "" (arXiv:) 
  * Also contained in the "PaperAndPresentation" folder is the thesis paper, conference paper, and presentation of the research.

# Usage
For data used in my experiments:
  * All datasets can be found in **Data/MIDI** and **Labels** (with example training data stored in **Images**), the PDF dataset of analyses is available upon request.
  * My most recent pre-trained weights can be found in **Weights/**.

**NOTE:** these folders should be placed in the **same** folder as "main.py". For folder existing conflicts, simply merge the directories.

In main.py, the "main" function acts as the controller for the model, where calls to train the model, create a prediction, and all other functions are called. One may also call these functions from an external script ("from main import get_total_duration", etc.).

To choose an operation or series of operations for the model to perform, simply edit the main function before running. Examples of all function calls can be seen commented out within main.

A demo of the full prediction system can be found [here](http://danielszelogowski.com/thesis/demo.php).

# Bugs/Features
Bugs are tracked using the GitHub Issue Tracker.

Please use the issue tracker for the following purpose:
  * To raise a bug request; do include specific details and label it appropriately.
  * To suggest any improvements in existing features.
  * To suggest new features or structures or applications.
