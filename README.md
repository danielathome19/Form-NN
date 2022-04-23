# About
Form-NN is a deep learning system (comprised of hybrid Neural Network-Decision Tree architectures, TreeGrad and Bi-LSTM-Tree) trained to perform both form and part/phrase analysis of classical music. The system also provides a new dataset that provides the form classification and timestamp-based analysis for 200 unique pieces of classical music.

To find out more, check out the provided research paper:
  * "Deep Learning for Musical Form: Recognition and Analysis" (DOI: coming soon) 
  * Also contained in the "PaperAndPresentation" folder is the thesis paper, conference paper, and presentation of the research.

# Usage
See:
  * http://danielszelogowski.com/thesis/demo.php for a live demo of the full prediction system.
  * https://github.com/danielathome19/Form-NN/releases for a downloadable local demo of the prediction system.

For data used in my experiments:
  * All datasets can be found in **Data/MIDI** and **Labels** (with example training data stored in **Images**), the PDF dataset of analyses is available upon request.
  * My most recent pre-trained weights can be found in **Weights/**.

**NOTE:** these folders should be placed in the **same** folder as "main.py". For folder existing conflicts, simply merge the directories.

In main.py, the "main" function acts as the controller for the model, where calls to train the model, create a prediction, and all other functions are called. One may also call these functions from an external script ("from main import get_total_duration", etc.).

To choose an operation or series of operations for the model to perform, simply edit the main function before running. Examples of all function calls can be seen commented out within main.

# Bugs/Features
Bugs are tracked using the GitHub Issue Tracker.

Please use the issue tracker for the following purpose:
  * To raise a bug request; do include specific details and label it appropriately.
  * To suggest any improvements in existing features.
  * To suggest new features or structures or applications.
