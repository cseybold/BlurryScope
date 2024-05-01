import os
import shutil
import random

def create_train_test_split(data_dir, output_dir, test_size=0.2, random_seed=2003):

    # Create output directories
    train_dir = os.path.join(output_dir, 'training_data')
    test_dir = os.path.join(output_dir, 'testing_data')
    sub_dirs = ['0', '1', '2', '3']
    for x in sub_dirs:
        if not os.path.exists(f"{train_dir}/{x}"):
            os.makedirs(f"{train_dir}/{x}")
        if not os.path.exists(f"{test_dir}/{x}"):
            os.makedirs(f"{test_dir}/{x}")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(('.tif')):
                diff, _ = os.path.splitext(filename)
                score = filename.rsplit('_', 1)[1].split('.')[0]
                slide = diff.split('_')[0]
                if slide == 'BR1141a' or slide == 'BR1202a':
                    shutil.copy(os.path.join(root, filename), os.path.join(test_dir, score))
                else:
                    shutil.copy(os.path.join(root, filename), os.path.join(train_dir, score))


main_data_dir = r"C:\Users\ammic\Documents\preDataTIF"
output_data_dir = "BlurScoreDataTwoTestTIF"

create_train_test_split(main_data_dir, output_data_dir)
print("Data split completed successfully!")
