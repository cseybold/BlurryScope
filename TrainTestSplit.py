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

    # Collect samples by sample name
    image_list = [[],[],[],[]]
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(('.png')):
                # diff = filename.rsplit('_', 1)[0]
                slide = filename.split('_')[0]
                core = filename.split('_')[2]
                diff = f"{slide}_{core}"
                score = int(filename.rsplit('_', 3)[2])
                # print(f'{filename}__{diff}__{score}')
                if score == 0:
                    if diff not in image_list[0]:
                        image_list[0].append(diff)
                if score == 1:
                    if diff not in image_list[1]:
                        image_list[1].append(diff)
                if score == 2:
                    if diff not in image_list[2]:
                        image_list[2].append(diff)
                if score == 3:
                    if diff not in image_list[3]:
                        image_list[3].append(diff)

    test_samples = [[],[],[],[]]
    train_samples = [[], [], [], []]
    random.seed(random_seed)
    for i in range(4):
        random.shuffle(image_list[i])
        num_test_samples = int(test_size * len(image_list[i]))
        test_samples[i] = image_list[i][:num_test_samples]
        train_samples[i] = image_list[i][num_test_samples:]

    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(('.tif')):
                # diff, _ = os.path.splitext(filename)
                slide = filename.split('_')[0]
                core = filename.split('_')[2]
                diff = f"{slide}_{core}"
                score = filename.rsplit('_', 1)[1].split('.')[0]
                # score = filename.split('_')[3]
                # print(f'{filename}__{diff}__{score}')
                if int(score) == 0:
                    if diff in train_samples[0]:
                        shutil.copy(os.path.join(root, filename), os.path.join(train_dir, score))
                    if diff in test_samples[0]:
                        shutil.copy(os.path.join(root, filename), os.path.join(test_dir, score))
                if int(score) == 1:
                    if diff in train_samples[1]:
                        shutil.copy(os.path.join(root, filename), os.path.join(train_dir, score))
                    if diff in test_samples[1]:
                        shutil.copy(os.path.join(root, filename), os.path.join(test_dir, score))
                if int(score) == 2:
                    if diff in train_samples[2]:
                        shutil.copy(os.path.join(root, filename), os.path.join(train_dir, score))
                    if diff in test_samples[2]:
                        shutil.copy(os.path.join(root, filename), os.path.join(test_dir, score))
                if int(score) == 3:
                    if diff in train_samples[3]:
                        shutil.copy(os.path.join(root, filename), os.path.join(train_dir, score))
                    if diff in test_samples[3]:
                        shutil.copy(os.path.join(root, filename), os.path.join(test_dir, score))


main_data_dir = r"C:\Users\ammic\Documents\preDataTIF"
output_data_dir = "BlurScoreDataTIF"

create_train_test_split(main_data_dir, output_data_dir)
print("Data split completed successfully!")
