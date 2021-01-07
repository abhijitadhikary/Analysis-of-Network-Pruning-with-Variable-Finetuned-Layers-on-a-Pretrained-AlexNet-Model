'''
Author: Abhijit Adhikary
Email: u7035746@anu.edu.au
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import torch
from torch import nn
from torchvision import models

emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# display an image
def imshow(im, title='', figsize=(4, 4), cmap='viridis'):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(im, cmap=cmap)
    plt.plot()
    plt.show()

def save_image_array(image_array, path):

    for current_index, current_image in enumerate(image_array):
        image_filename = f'{current_index}.jpg'
        current_image_path = os.path.join(path, image_filename)
        cv2.imwrite(current_image_path, current_image)

# prepare the dataset from the original SFEW dataset (675)
def split_dataset(source_dataset_path='Subset For Assignment SFEW', destinaiton_dataset_path='SFEW Modified'):
    if os.path.exists(destinaiton_dataset_path) is False:

        dir_level_1 = ['train', 'test', 'validation']
        dir_level_2 = emotion_list

        for dir_l_1 in dir_level_1:
            level_1 = os.path.join(destinaiton_dataset_path, dir_l_1)
            for dir_l_2 in dir_level_2:
                level_2 = os.path.join(level_1, dir_l_2)
                if not os.path.exists(level_2):
                    os.makedirs(level_2, exist_ok = True)

        destination_train_path = os.path.join(destinaiton_dataset_path, 'train')
        destination_validation_path = os.path.join(destinaiton_dataset_path, 'validation')
        destination_test_path = os.path.join(destinaiton_dataset_path, 'test')

        np.random.seed(1)

        category_names = os.listdir(source_dataset_path)
        num_categories = len(category_names)

        train_percentage = 0.6
        validation_percentage = 0.5

        for _, current_category in enumerate(category_names):
            source_category_path = os.path.join(source_dataset_path, current_category)
            image_filenames = os.listdir(source_category_path)

            image_array = []
            for index, current_image_filename in enumerate(image_filenames):
                current_image_path = os.path.join(source_category_path, current_image_filename)
                current_image = cv2.imread(current_image_path, 1)
                image_array.append(current_image)
            image_array = np.array(image_array, dtype=np.uint8)

            # extract the training set
            mask_1 = np.random.rand(len(image_array)) < train_percentage
            train_array = image_array[mask_1]
            remaining_array = image_array[~mask_1]

            # extract the test set
            mask_2 = np.random.rand(len(remaining_array)) < validation_percentage
            validation_array = remaining_array[mask_2]
            test_array = remaining_array[~mask_2]

            current_train_path = os.path.join(destination_train_path, current_category)
            current_validation_path = os.path.join(destination_validation_path, current_category)
            current_test_path = os.path.join(destination_test_path, current_category)

            save_image_array(train_array, current_train_path)
            save_image_array(validation_array, current_validation_path)
            save_image_array(test_array, current_test_path)
        print('Dataset Created')
    else:
        print('Dataset already exists')

# calculate the Specific Person Independent Values (SPI)
def get_SPI_calculations(y_pred, y, num_classes, num_samples):
    pred_array = np.empty([num_classes, num_samples], dtype=object)

    y_pred = np.squeeze(np.array(y_pred))
    y = np.squeeze(np.array(y))

    for current_sample in range(num_samples):
        for current_class in range(num_classes):
            predicted_class = y_pred[current_sample]
            actual_class = y[current_sample]

            if (predicted_class == current_class) and (actual_class == current_class):
                pred_array[current_class][current_sample] = 'tp'
            elif (predicted_class == current_class) and (actual_class != current_class):
                pred_array[current_class][current_sample] = 'fp'
            elif (predicted_class != current_class) and (actual_class == current_class):
                pred_array[current_class][current_sample] = 'fn'
            elif (predicted_class != current_class) and (actual_class != current_class):
                pred_array[current_class][current_sample] = 'tn'

    accuracy_array = np.zeros((num_classes))
    precicion_array = np.zeros((num_classes))
    recall_array = np.zeros((num_classes))
    specificity_array = np.zeros((num_classes))

    for current_class in range(num_classes):
        class_array = pred_array[current_class]
        tp = np.sum(class_array == 'tp')
        tn = np.sum(class_array == 'tn')
        fp = np.sum(class_array == 'fp')
        fn = np.sum(class_array == 'fn')

        if (tp + fp + fn + tn) != 0:
            accuracy_array[current_class] = (tp + tn) / (tp + fp + fn + tn)
        else:
            accuracy_array[current_class] = 0

        if (tp + fp) != 0:
            precicion_array[current_class] = tp / (tp + fp)
        else:
            precicion_array[current_class] = 0

        if (tp + fn) != 0:
            recall_array[current_class] = tp / (tp + fn)
        else:
            recall_array[current_class] = 0

        if (tn + fp) != 0:
            specificity_array[current_class] = tn / (tn + fp)
        else:
            specificity_array[current_class] = 0

    return accuracy_array, precicion_array, recall_array, specificity_array

# generate a confusion matrix given actual and predicted labels
def get_confusion_matrix(y_pred, y, num_classes, num_samples):

    y_pred = np.squeeze(np.array(y_pred))
    y = np.squeeze(np.array(y))

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for current_sample in range(num_samples):
        current_class = y[current_sample]
        predicted_class = y_pred[current_sample]
        confusion_matrix[current_class][predicted_class] += 1

    return confusion_matrix

# plot the confusion matrix using matplotlib (heatmap)
def display_confusion_matrix(confusion_matrix, title=''):
    plt.figure(figsize=(7, 7))
    plt.title(title)
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.xticks(range(7), emotion_list)
    plt.yticks(range(7), emotion_list)
    plt.colorbar()
    plt.show()

# function to print the calculated SPI scores
def print_SPI_scores(SPI_parameters):
    classwise_accuracy_array, precicion_array, recall_array, specificity_array, num_classes = SPI_parameters
    row_headers = ['Emotion', 'Precicion', 'Recall', 'Specificity']

    print('_________________________________________________________________')
    print(row_headers[0], end='\t\t')
    for i in range(num_classes):
        print(emotion_list[i], end='\t')
    print('\n_________________________________________________________________')

    print(row_headers[1], end='\t')
    for i in range(num_classes):
        print(f'{precicion_array[i]:.2f}', end='\t')
    print('')

    print(row_headers[2], end='\t\t')
    for i in range(num_classes):
        print(f'{recall_array[i]:.2f}', end='\t')
    print('')

    print(row_headers[3], end='\t')
    for i in range(num_classes):
        print(f'{specificity_array[i]:.2f}', end='\t')
    print('\n')

# freeze a specified amount of layers, weight updates won't be performed upto the n'th layer
def freeze_layers(model, num_layers_to_freeze=0):
    if (num_layers_to_freeze <= 0) or (num_layers_to_freeze) > 7:
        return model
    layers_with_grad = [0, 3, 6, 8, 10, 14, 17] # 19

    num_layers = 0
    layer_index = 0
    for child in model.children():
        for current_layer in child.children():
            if layer_index in layers_with_grad:
                for parameters in current_layer.parameters():
                    parameters.requires_grad = False
                num_layers += 1
            layer_index += 1

            if num_layers == num_layers_to_freeze:
                break
    return model

# print the layers which are in active state (weights are being updated)
def print_active_layers(model):
    layers_with_grad = [0, 3, 6, 8, 10, 14, 17] # 19

    active_layers = [False, False, False, False, False, False, False]

    num_layers = 0
    layer_index = 0
    for child in model.children():
        for current_layer in child.children():
            if layer_index in layers_with_grad:
                for parameters in current_layer.parameters():
                    if parameters.requires_grad == True:
                        active_layers[num_layers] = True
                num_layers += 1

            layer_index += 1
    print('Active Layers:')
    for index in range(len(active_layers)):
        print(f'Layer {index}: {active_layers[index]}')

# save the current state of the model
def save_checkpoint(checkpoint_parameters):
    model, model_name, current_epoch, num_epochs, timestamp, comment, num_layers_to_freeze = checkpoint_parameters
    checkpoint = {'model_state_dict': model.state_dict(),
                  'model_name': model_name,
                  'current_epoch': current_epoch,
                  'num_epochs': num_epochs,
                  'timestamp': timestamp,
                  'comment': comment,
                  'num_layers_to_freeze': num_layers_to_freeze
                 }

    torch.save(checkpoint, f'checkpoints/{timestamp}__{comment}_ep_{current_epoch}.pth')

# load a previously saved model
def load_checkpoint(filepath, num_categories):
    checkpoint = torch.load(filepath)
    if checkpoint['model_name'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_categories, bias=True)
    else:
        print("Unknown model")

    model.load_state_dict(checkpoint['model_state_dict'])
    model_name = checkpoint['model_name']
    current_epoch = checkpoint['current_epoch']
    num_epochs = checkpoint['num_epochs']
    timestamp = checkpoint['timestamp']
    comment = checkpoint['comment']
    num_layers_to_freeze = checkpoint['num_layers_to_freeze']
    freeze_layers(num_layers_to_freeze)

    return model, model_name, current_epoch, num_epochs, timestamp, comment, num_layers_to_freeze


# get the angle between two vector in degrees
def get_vector_angle(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    u_dot_v = np.dot(u, v)

    cosTheta = np.divide(u_dot_v, np.multiply(norm_u, norm_v))
    # cosTheta = u_dot_v / (norm_u * norm_v)

    theta_radian = np.arccos(cosTheta)
    theta_degree = np.rad2deg(theta_radian)

    # if a NaN value is encountered
    if np.isnan(theta_degree):
        theta_degree = 1000

    return theta_degree

# prune neurons givel degree and standard deviation threshold
def prune_neurons(model, hidden_activation_vector_np, degree_low=15, degree_high=165, std_threshold=0):
    # normalize values between [-0.5, 0.5]
    lowerbound = -0.5
    upperbound = 0.5
    min_value = hidden_activation_vector_np.min()
    max_value = hidden_activation_vector_np.max()
    mean_value = hidden_activation_vector_np.max()
    std_value = hidden_activation_vector_np.std()
    upper_limit = mean_value + std_value
    lower_limit = mean_value - std_value

    hidden_activation_vector_np = (hidden_activation_vector_np - min_value) * (upperbound - lowerbound) / (
                max_value - min_value) + lowerbound

    num_samples_deg = hidden_activation_vector_np.shape[0]
    num_neurons_deg = hidden_activation_vector_np.shape[1]

    prune_list = []
    norm_mean = np.linalg.norm(hidden_activation_vector_np, axis=0).mean()
    norm_std = np.linalg.norm(hidden_activation_vector_np, axis=0).std()
    upper_limit = norm_mean + norm_std * std_threshold
    lower_limit = norm_mean - norm_std * std_threshold

    for row in tqdm(range(num_neurons_deg)):
        neuron_1 = hidden_activation_vector_np[:, row]
        if (np.linalg.norm(neuron_1) < lower_limit) or (np.linalg.norm(neuron_1) > upper_limit):
            for col in range(num_neurons_deg):
                if row != col:
                    neuron_2 = hidden_activation_vector_np[:, col]
                    if (np.linalg.norm(neuron_2) < lower_limit) or (np.linalg.norm(neuron_2) > upper_limit):
                        # calculate the angle between weight vectors
                        degree = get_vector_angle(neuron_1, neuron_2)

                        if (degree < degree_low) or (degree > degree_high):
                            # to ingnore the order of the neurons
                            if col < row:
                                if [col, row] not in prune_list:
                                    prune_list.append([col, row])
                            else:
                                if [row, col] not in prune_list:
                                    prune_list.append([row, col])

    # set the weights of the corresponding neurons
    for neuron_1, neuron_2 in prune_list:
        # add the weight of the 2nd neuron to the 1st neuron
        # model.fc1.weight[neuron_1] = (model.fc1.weight[neuron_1] + model.fc1.weight[neuron_2]) / 2.0
        model.classifier[4].weight[neuron_1] = (
                    model.classifier[4].weight[neuron_1] + model.classifier[4].weight[neuron_2])

        # set the weight of the 2nd neuron to zero
        model.classifier[4].weight[neuron_2] = torch.zeros(model.classifier[4].weight[0].data.shape,
                                                           requires_grad=False)

    num_prune = 0

    # if number of pruned neurons is greater than 0, calculate the length
    try:
        num_prune = len(np.unique(np.array(prune_list)[:, 1]))
    except:
        pass
    return model, prune_list, num_prune