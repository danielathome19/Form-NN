# About
Form-NN is a deep learning system (comprised of hybrid Neural Network-Decision Tree architectures, TreeGrad and Bi-LSTM-Tree) trained to perform both form and part/phrase analysis of classical music. The system also provides a new dataset that provides the form classification and timestamp-based analysis for 200 unique pieces of classical music.

To find out more, check out the provided research paper:
  * "Deep Learning for Musical Form: Recognition and Analysis" (DOI: coming soon) 
  * Also contained in the ["PaperAndPresentation"](https://github.com/danielathome19/Form-NN/tree/master/PaperAndPresentation) folder is the thesis paper, conference paper, and presentation of the research.
  * The thesis defense can be watched at https://youtu.be/2ZM5jz5gows.

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

# Contribution
The dataset currently only has a small number of pieces of music fully analyzed by timestamp. If you would like to contribute to this dataset by providing analyses, please contact me with the following:
  * What piece(s) you would like to analyze — I can send you the PDF of the sheet music or you can find any of the scores on IMSLP for free.
  * Your musical background — the dataset needs to follow standardized analytical conventions; currently, these are based very closely on the analysis techniques presented in Douglass Green's **Form in Tonal Music: An Introduction to Analysis, 2nd Ed.** [ISBN: 0030202868, ISBN13: 9780030202865](https://www.thriftbooks.com/w/form-in-tonal-music-an-introduction-to-analysis_douglass-m-green/264527/item/527721/?mkwid=qmNYahkX%7cdc&pcrid=11558858230&pkw=&pmt=be&slid=&product=527721&plc=&pgrid=3970769304&ptaid=pla-1101002864651&utm_source=bing&utm_medium=cpc&utm_campaign=Bing+Shopping+%7c+Arts+&+Photography&utm_term=&utm_content=qmNYahkX%7cdc%7cpcrid%7c11558858230%7cpkw%7c%7cpmt%7cbe%7cproduct%7c527721%7cslid%7c%7cpgrid%7c3970769304%7cptaid%7cpla-1101002864651%7c&msclkid=bb79bd7f27dd11661d09dd7bdc3322e7#idiq=527721&edition=2332832). You should have at least taken an undergraduate *Form and Analysis* class or have an equal background in music theory.
  * If you need, I can provide a draft of annotation guidelines and/or my own example analyses. This will be added as a document later on when fully prepared.

After you complete the analysis of the sheet music, use the existing label file for the MIDI(s) in the dataset to start labeling the timestamps with your annotations. Feel free to email me with your analyzed sheet music for corrections. Refer to Appendices A and B and Chapter 3 in the thesis as needed.

When you are finished with the analysis, email me the finished analyzed score and label file, or send me the score and push the label file to the repository.

# Bugs/Features
Bugs are tracked using the GitHub Issue Tracker.

Please use the issue tracker for the following purpose:
  * To raise a bug request; do include specific details and label it appropriately.
  * To suggest any improvements in existing features.
  * To suggest new features or structures or applications.
