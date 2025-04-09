
# Meat Freshness Detection

## Project Overview
This project utilizes a modified ResNet18 model to classify images of meat into three categories based on their freshness: Fresh, Half-Fresh, and Spoiled. It monitors a specified folder for new images, processes them in real-time, and displays the classification results through a Gradio interface.

## Features
- **Real-time Image Classification**: Automatically classifies images as they are added to the monitored folder.
- **Interactive Web Interface**: View predictions and corresponding images through a user-friendly Gradio interface.
- **Automated Monitoring**: Uses the `watchdog` library to watch for new images in a specified directory.

## Installation
To run this project, you'll need Python 3.6 or later and the following packages:
- torch
- torchvision
- PIL
- watchdog
- gradio

Clone this repository to your local machine:
```bash
git clone https://github.com/2abet/Meat-Grader.git
cd Meat-Grader
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage
To start the meat freshness detection, navigate to the project directory and run:
```bash
python prediction.py
```

Make sure to update the `folder_path` variable in the script to point to the directory you want to monitor.

## Contributing
Contributions to this project are welcome! Here's how you can contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your_feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your_feature`).
5. Open a pull request.

## License
Specify the license under which your project is made available. (e.g., MIT, GPL, Apache, etc.)

## Contact
Your Name – Akinyemi Arabambi

Project Link: [https://github.com/2abet/Meat-Grader](https://github.com/2abet/Meat-Grader)
