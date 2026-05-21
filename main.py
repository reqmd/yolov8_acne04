import subprocess
import os

DATA_PATH = 'data'
root_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    print('First step: download and processing data')
    if not os.path.exists(os.path.join(DATA_PATH, 'Annotations')) and not os.path.exists(os.path.join(DATA_PATH, 'Patches')):
        subprocess.run(["python", "src/data/download_done_data.py"], check = True)
        subprocess.run(["python", "src/data/preprocessing_sliding_window.py"], check = True)
    else:
        print('Data already exists')
    print('=' * 60)

    train_answer = int(input('Skip train? 0 - No, 1 - Yes\n'))
    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir
    if train_answer == 0:
        print('Second step: train model on processed data')
        subprocess.run(["python", "scripts/train.py"], cwd=root_dir, env=env, check=True)
        print('=' * 60)

    print('Before testing the model, make sure that the test image is located in the data/Test folder')
    testing_answer = int(input('Need test model? 0 - No, 1 - Yes\n'))
    if testing_answer == 1:
        subprocess.run(["python", "scripts/test.py"], cwd=root_dir, env=env, check=True)

