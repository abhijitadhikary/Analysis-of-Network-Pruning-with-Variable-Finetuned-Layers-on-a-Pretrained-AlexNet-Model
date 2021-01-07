'''
Author: Abhijit Adhikary
Email: u7035746@anu.edu.au
'''

from datetime import datetime
from train_test import *
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

def run_model():
    # conditions and hyper parameters
    train_condition = True
    load_and_test_condition = False
    load_path = 'checkpoints/top_test.pth'

    prune_condition = False
    std_threshold = 3
    degree_threshold = 15
    save_prune_stats = False
    test_pruned_performance = True

    # paths to datasets
    dataset_path = 'SFEW Modified'
    train_path = os.path.join(dataset_path, 'train')
    validation_path = os.path.join(dataset_path, 'validation')
    test_path = os.path.join(dataset_path, 'test')

    # create the custom dataset from the original dataset
    # convert to 224 x 224 x 3 for training with AlexNet
    split_dataset(source_dataset_path='Subset For Assignment SFEW', \
                  destinaiton_dataset_path=dataset_path)

    # random rotation, crops and normalization for training, validation, and test sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    # create the datasets
    training_dataset = datasets.ImageFolder(train_path, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(validation_path, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_path, transform=testing_transforms)

    num_train = len(training_dataset)
    num_validation = len(validation_dataset)
    num_test = len(testing_dataset)
    train_batch_size = num_train
    num_categories = 7 # number of categories of the SFEW datasets (emotions)

    # create dataloaders
    # might need to reduce the batch_size if the system does not have a high performance GPU
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True, num_workers=7)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=num_validation, num_workers=7)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=num_test, num_workers=7)

    # initiate the model
    model = models.alexnet(pretrained=True)
    model_name = 'alexnet'
    model.classifier[6] = nn.Linear(in_features=4096, out_features=num_categories, bias=True)

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    # perform pruning
    if prune_condition:
        print_prune_statements = False
        degree_list = [17.5, 15, 12.5, 10, 7.5]
        std_threshold_list = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1]

        # perform grid search for pruning
        for std_threshold in std_threshold_list:
            for degree_threshold in degree_list:
                if print_prune_statements: print('Pruning Started')

                # load the pretrained mdoel
                model, model_name, current_epoch, num_epochs, timestamp, comment, num_layers_to_freeze = load_checkpoint(
                    load_path, num_categories)
                if print_prune_statements: print('Pretrained Model Loaded')

                # extract the hidden layer (7th in Pytorch's AlexNet model)
                outputs = []
                def hook(module, input, output):
                    outputs.append(output)
                model.classifier[4].register_forward_hook(hook)
                prune_batch_size = 1
                prune_loader = torch.utils.data.DataLoader(training_dataset, \
                                batch_size=prune_batch_size, shuffle=True, num_workers=7)
                model = model.to(device)

                # extract the hidden outputs
                for index, prune_batch in enumerate(prune_loader):
                    images, labels = prune_batch
                    images, labels = images.to(device), labels.to(device)
                    out = model(images)

                num_samples = len(outputs)
                num_neurons = outputs[0].shape[1]
                hidden_layer_outputs = np.zeros((num_samples, num_neurons))

                # convert hidden outputs to numpy array
                for index in range(num_samples):
                    hidden_layer_outputs[index] = np.array(outputs[index].cpu().detach())

                if print_prune_statements:print('Hidden Layer (4096 x 4096) Outputs Extracted')

                torch.cuda.empty_cache()

                # perform pruning
                model, prune_list, num_prune = prune_neurons(model, hidden_layer_outputs, \
                                                             degree_low=0+degree_threshold, degree_high=180-degree_threshold,
                                                             std_threshold=std_threshold)
                if print_prune_statements:print(f'Pruned Neurons: {num_prune}')

                # save the pruned statistics
                if save_prune_stats:
                    np.save(f'prune_parameters/num_prune_std_thresh_{std_threshold}.npy', num_prune)
                    np.save(f'prune_parameters/prune_list_std_thresh_{std_threshold}.npy', prune_list)

                    learning_rate = 0.001
                    num_layers_to_freeze = 6
                    now = datetime.now()
                    timestamp = now.strftime("%Y_%m_%d %H_%M_%S")
                    comment = f'lr={learning_rate}, num_freeze={num_layers_to_freeze}, run=3'

                    checkpoint_parameters = model, 'alexnet', 50, 50, timestamp, comment, 0
                    save_checkpoint(checkpoint_parameters)

                # test the network after pruning
                if test_pruned_performance:
                    test_parameters = model, device, test_loader, criterion
                    test_loss, test_accuracy, SPI_parameters, confusion_matrix = test_model(test_parameters)
                    print(f'\nD.Offset:\t{degree_threshold}\tSTD Offset:\t{std_offset}', end='\t')
                    print(f'Pruned:\t{num_prune}\tAccuracy:\t{test_accuracy:.2f} %')
                    # print(f"Pruned Test Loss: {test_loss:6f}\tAccuracy: {test_accuracy:.2f} %")

                    # print_SPI_scores(SPI_parameters)
                    # display_confusion_matrix(confusion_matrix, title='Test')

    # load and test a pretrained model
    if load_and_test_condition:
        model, model_name, current_epoch, num_epochs, timestamp,\
            comment, num_layers_to_freeze = load_checkpoint(load_path, num_categories)
        test_parameters = model, device, test_loader, criterion
        test_loss, test_accuracy, SPI_parameters, confusion_matrix = test_model(test_parameters)
        print(f"Test Loss: {test_loss:6f}\tAccuracy: {test_accuracy:.2f} %")

        print_SPI_scores(SPI_parameters)
        display_confusion_matrix(confusion_matrix, title='Test')

    # train model
    if train_condition:
        load_model = False
        num_train_runs = 1
        # for num_layers_to_freeze in range(7, -1, -1):
        for current_run in range(num_train_runs):
            # load Pytorch's pretrained AlexNet model
            model = models.alexnet(pretrained=False)
            model_name = 'alexnet'
            model.classifier[6] = nn.Linear(in_features=4096, out_features=num_categories, bias=True)

            num_layers_to_freeze = 0 # max 7
            num_epochs = 1
            epoch_start = 1
            learning_rate = 0.001
            validation_interval = 1
            now = datetime.now()
            timestamp = now.strftime("%Y_%m_%d %H_%M_%S")
            comment = f'lr={learning_rate}, num_freeze={num_layers_to_freeze}, run={current_run}'
            save_checkpoint_condition = True

            # resume training from a previous checkpoint
            if load_model:
                load_path = 'checkpoints/2020_05_30 19_48_41__lr=0.001, num_freeze=5_ep_50.pth'
                model, model_name, current_epoch, num_epochs, timestamp, comment, num_layers_to_freeze = load_checkpoint(load_path, num_categories)
                epoch_start = current_epoch + 1
                num_epochs = num_epochs + 30

            # perform regularization
        #     weight_decay = 0.05
        #     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        #     decay_rate = 0.96
        #     learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
            learning_rate_scheduler = None

            # initiate SummaryWriter for visualizing in TensorBoard
            writer = SummaryWriter(comment=comment)

            train_parameters = model, criterion, num_epochs, device, train_loader, validation_interval, validation_loader, \
                writer, save_checkpoint_condition,  model_name, epoch_start, learning_rate_scheduler, optimizer,  timestamp, \
                comment, num_layers_to_freeze

            model = freeze_layers(model, num_layers_to_freeze)
            train_model(train_parameters)

            # test the trained model
            print('--------------------- Final Test ---------------------')
            print(comment)
            test_parameters = model, device, test_loader, criterion
            test_loss, test_accuracy, SPI_parameters, confusion_matrix = test_model(test_parameters)
            print(f"Test Loss: {test_loss:6f}\tAccuracy: {test_accuracy:.2f} %")

            print_SPI_scores(SPI_parameters)
            display_confusion_matrix(confusion_matrix, title='Test')

if __name__=='__main__':
    run_model()