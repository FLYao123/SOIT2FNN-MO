This repository contains supplementary code accompanying the following paper:

Yao, F., Zhao, W., Forshaw, M., & Song, Y. (2025).
A self-organizing interval type-2 fuzzy neural network for multi-step time series prediction.
Applied Soft Computing, 113221.

⚠️ Note

To maintain consistency with academic standards, certain symbols in the published manuscript have been adjusted. As a result, there may be discrepancies between the notation used in the code and that in the paper.

We have added comments in the code to highlight these differences.

Additional comments are included in key sections to assist readers in understanding the implementation.

Due to time constraints, some sections may contain minimal annotation. We recommend referring to the published paper for further clarification.

🚀 How to Run the Code

🔧 Training
To begin training the model:

Run Main_training.m in the root directory.

By default, the model is trained on a Chaotic time series dataset.

An alternative Microgrid dataset (including price and unmet power data) is also provided in the repository. You can switch datasets by following the in-code comments.

📊 Testing
To evaluate the trained model:

Run Main_test.m.

By default, testing is performed on the Chaotic time series dataset.

Pre-trained weights for this dataset are available in the Weights folder.

You may also retrain the model using either the Chaotic or Microgrid datasets before testing, if desired.

📁 Additional Reference Code

While this repository focuses on the implementation of the IT2FNN-MO model (as demonstrated in Main_training.m and Main_test.m), we also include reference drafts of the following models for comparative or exploratory purposes:

IT2FNN-SW — located in the IT2FNNP1 directory

IT2FNN-PM — located in the IT2FNNP2 directory

These supplementary directories contain early-stage versions and may not be fully annotated.


