# Automatic Logo Detection - Phase 1

## Description
A [gradio](https://www.gradio.app/guides/quickstart) application allowing users to detect potential logos given an input image. It performs zero-shot object detection using the [OWLv2 Model](https://huggingface.co/docs/transformers/main/model_doc/owlv2).

## Installation and Usage

1. **Install Python 3.9+ on your machine.**

2. **Create and activate a python virtual enviornment (optional).**
    - Follow this [link](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for instruction on how to create one.

3. **Run `pip install -r requirements.txt` on the command line or powershell to install the necessary dependencies.**
    - Note: If you have access to a GPU, please install the correct version of `pytorch` from [this](https://pytorch.org/get-started/locally/) link according to your CUDA version.

4. **Run `python app.py` on the command line or powershell to launch the GUI locally. Click on the given link to open the GUI in your browser.**
    - Note: This will take some time to run the first time as it needs to download the OWLv2 model.

5. **Follow the instructions given at the bottom of the application.**

6. **If you have a GPU and the correct version of pytorch installed, the app will automatically use the GPU for detection.**
    - With GPU, detection time per image is approximately 4 seconds.
    - Without GPU, detection time per image is approximately 50 seconds.

6. **Experiment with different images.**


## Credits
Credit goes to the authors of OWLv2 and Gradio. 