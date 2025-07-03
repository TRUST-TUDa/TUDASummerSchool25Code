import torch
import numpy as np
import datetime
import os
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from pytorch_lightning import seed_everything
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from IPython.display import display, Markdown
import hashlib
from TUDASummerSchool25Code.solutions import SOLUTIONS, HASHES, MAX_CHR
from TUDASummerSchool25Code.ModelUtils import test, test_split, model_dist_norm

TRAINING_BATCH_SIZE = 64
STD_DEV = torch.from_numpy(np.array([0.2023, 0.1994, 0.2010]))
MEAN = torch.from_numpy(np.array([0.4914, 0.4822, 0.4465]))

def sort_samples_by_labels(dataset):
        cifar_classes = defaultdict(list)
        all_training_images = []
        for ind, (_, label) in tqdm(enumerate(dataset)):
            cifar_classes[label].append(ind)
            all_training_images.append(ind)
        all_labels = sorted(list(cifar_classes.keys()))
        cifar_classes = {label: np.array(images) for label, images in cifar_classes.items()}
        return cifar_classes, all_labels, np.array(all_training_images)

def format_for_logging(text):
    text = str(text).split('\n')
    current_time = str(datetime.datetime.now())
    text = ['{0}: {1}'.format(current_time, line) for line in text]
    text = '\n'.join(text)
    return text

def print_timed(text):
    text = format_for_logging(text)
    print(text)

def batchify(source, batch_size, total_number):
    source = [x for x in source]
    result = []
    for batch_start in range(0, total_number, batch_size):
        batch_end = min(total_number, batch_start + batch_size)
        to_add = source[batch_start:batch_end]
        assert len(to_add) > 0
        if isinstance(to_add[0], tuple):
            tensor = torch.stack([p[0] for p in to_add])
            label = torch.LongTensor([p[1] for p in to_add])
            to_add = [tensor, label]
        else:
            to_add = torch.stack(to_add)
        result.append(to_add)
    return result

class MyDataLoader:

    def __init__(self, all_data, indices, batch_size):
        assert all_data is not None
        assert indices is not None
        assert batch_size is not None
        data_x = [all_data.data[x] for x in indices]
        data_x = [MyDataLoader.transform_images(x, all_data) for x in data_x]
        data_x = torch.stack(data_x, axis=0)
        data_y = [all_data.targets[x] for x in indices]
        if isinstance(data_y, list):
            data_y = torch.tensor(data_y)
        total_number_of_samples_for_this_client = len(indices)
        batches_X = batchify(data_x, batch_size, total_number_of_samples_for_this_client)
        
        batches_y = batchify(data_y, batch_size, total_number_of_samples_for_this_client)
        self.batches = list(zip(batches_X, batches_y))
    
    @staticmethod
    def transform_images(img, dataset):
        if torch.is_tensor(img):
            img = img.cpu().detach().numpy()
        img = Image.fromarray(img)
        img = dataset.transform(img)

        return img

    def __iter__(self):
        return self.batches.__iter__()

def create_client_distributions(total_client_number, iid_rate, samples_per_client, all_labels, train_data_by_labels,all_training_images):
        seed_everything(42)
        clients_main_labels = np.random.choice(all_labels, size=total_client_number)
        samples_of_main_class_per_client = int((1 - iid_rate) * samples_per_client)
        samples_of_all_classes_per_client = samples_per_client - samples_of_main_class_per_client
        print_timed(f'Samples from main class per client: {samples_of_main_class_per_client}')
        print_timed(f'Samples from all classes per client: {samples_of_all_classes_per_client}')
        indices_for_clients = []
        main_labels_dict = {}
        for client_index, main_label in enumerate(clients_main_labels):
            indices_of_current_client = -1 * np.ones(samples_per_client)
            indices_for_main_label = np.random.choice(indices_of_current_client.shape[0],
                                                      samples_of_main_class_per_client, replace=False)
            assert indices_for_main_label.shape[0] == samples_of_main_class_per_client
            indices_of_current_client[indices_for_main_label] = -2
            indices_for_other_labels = np.where(indices_of_current_client == -1)[0]
            indices_of_current_client[indices_for_main_label] = np.random.choice(train_data_by_labels[main_label], samples_of_main_class_per_client, replace=False)
            other_images = np.random.choice(all_training_images, samples_of_all_classes_per_client, replace=False)
            indices_of_current_client[indices_for_other_labels] = other_images
            indices_for_clients.append(indices_of_current_client.astype(int))
            main_labels_dict[client_index] = main_label
        print_timed(f'Main label for clients: {main_labels_dict} ')
        return indices_for_clients, main_labels_dict

def unnormalize_image(tensor, std_dev, mean):
    result = tensor.clone()
    for t, m, s in zip(result, mean, std_dev):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    if result.min() < 0:
        assert result.min() > - (10 ** -6)
        result -= result.min()
    return result.transpose(0,1).transpose(1,2)

def create_cluster_map_from_labels(expected_number_of_labels, clustering_labels):
    assert len(clustering_labels) == expected_number_of_labels

    clusters = defaultdict(list)
    for i, cluster in enumerate(clustering_labels):
        clusters[cluster].append(i)
    return {index: np.array(cluster) for index, cluster in clusters.items()}

def plot_grid(image_1, image_2, image_3, image_4):
    np_img_1 = image_1.transpose(0,1).transpose(1,2).numpy()
    np_img_2 = image_2.transpose(0,1).transpose(1,2).numpy()
    np_img_3 = image_3.transpose(0,1).transpose(1,2).numpy()
    np_img_4 = image_4.transpose(0,1).transpose(1,2).numpy()
    np_img_1 = (np_img_1 * 255).astype(np.uint8)
    np_img_2 = (np_img_2 * 255).astype(np.uint8)
    np_img_3 = (np_img_3 * 255).astype(np.uint8)
    np_img_4 = (np_img_4 * 255).astype(np.uint8)

    fig, axs = plt.subplots(2, 2, figsize=(8,8))

    axs[0, 0].imshow(np_img_1)
    axs[0, 1].imshow(np_img_2)
    axs[1, 0].imshow(np_img_3)
    axs[1, 1].imshow(np_img_4)
    plt.show()

def plot_image(image):
    np_img = image.transpose(0,1).transpose(1,2).numpy()
    plt.imshow((np_img * 255).astype(np.uint8))

def unnormalize(tensor):
    result = tensor.clone()
    for t, m, s in zip(result, MEAN, STD_DEV):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return result

def poison_dataset_DBA(benign_local_dataset, target_label, trigger_function, pdr=0.5, print_number_of_poisoned_images=False, batch_size=TRAINING_BATCH_SIZE):
    all_labels = []
    all_images = []
    desired_number_of_poisoned_images = 0
    actually_poisoned_images = 0
    for samples, labels in benign_local_dataset:
        samples, labels = samples.clone(), labels.clone()
        desired_number_of_poisoned_images += pdr * labels.shape[0]
        samples_to_poison = int(np.floor(desired_number_of_poisoned_images) - actually_poisoned_images)
        for i in range(samples_to_poison):
            labels[i] = target_label
            samples[i] = trigger_function(samples[i])
            actually_poisoned_images += 1
        all_labels.append(labels)
        all_images.append(samples)
    all_labels = torch.concat(all_labels)
    all_images = torch.concat(all_images)
    if print_number_of_poisoned_images:
        print(f'Poisoned {actually_poisoned_images} from {int(desired_number_of_poisoned_images)} images')
    return DataLoader(TensorDataset(all_images, all_labels), batch_size=batch_size, shuffle=False)


def visualize_model_predictions(dataset_to_use, model_to_test, classes, std_dev, mean, computation_device, batch_index_to_plot=2, show_labels=True):

    data_batch = list(dataset_to_use)[batch_index_to_plot]
    model_to_test.eval()
    data, labels = data_batch
    output = model_to_test(data.to(computation_device)).detach()
    pred = output.data.max(1)[1]
    pred = pred.cpu().numpy()
    unique_labels, counts = np.unique(pred, return_counts=True)
    distribution_map = {l: count for l, count in zip(unique_labels, counts)}
    print(f'Prediction Distribution: {distribution_map}')
    fig, axs = plt.subplots(1, 6, dpi=300)
    for x in range(6):
        axs[x].imshow(unnormalize_image(data[x*24].cpu(), std_dev=std_dev,mean=mean))
        title = ''
        if show_labels:
            title += f'Label: {classes[labels[x*24]]}\n'
        title += f'Prediction: {classes[pred[x*24]]}'
        axs[x].set_title(title, fontsize=4)
        axs[x].get_xaxis().set_visible(False)
        axs[x].get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()
    return 

def visualize_split_model_predictions(dataset_to_use, head, backbone, tail, classes, std_dev, mean, computation_device, batch_index_to_plot=2, show_labels=True):
    stored_device_head = None
    stored_device_back = None
    stored_device_tail = None
    if next(head.parameters()).device != computation_device:
        stored_device_head = next(head.parameters()).device
        head = head.to(computation_device)
    if next(backbone.parameters()).device != computation_device:
        stored_device_back = next(backbone.parameters()).device
        backbone = backbone.to(computation_device)
    if next(tail.parameters()).device != computation_device:
        stored_device_tail = next(tail.parameters()).device
        tail = tail.to(computation_device)

    data_batch = list(dataset_to_use)[batch_index_to_plot]
    head.eval()
    backbone.eval()
    tail.eval()
    data, labels = data_batch
    output = tail(backbone(head(data.to(computation_device)))).detach()
    pred = output.data.max(1)[1]
    pred = pred.cpu().numpy()
    unique_labels, counts = np.unique(pred, return_counts=True)
    distribution_map = {l: count for l, count in zip(unique_labels, counts)}
    print(f'Prediction Distribution: {distribution_map}')
    fig, axs = plt.subplots(1, 6, dpi=300)
    for x in range(6):
        axs[x].imshow(unnormalize_image(data[x*24].cpu(), std_dev=std_dev,mean=mean))
        title = ''
        if show_labels:
            title += f'Label: {classes[labels[x*24]]}\n'
        title += f'Prediction: {classes[pred[x*24]]}'
        axs[x].set_title(title, fontsize=4)
        axs[x].get_xaxis().set_visible(False)
        axs[x].get_yaxis().set_visible(False)
    if stored_device_head != None:
        head = head.to(stored_device_head)
    if stored_device_back != None:
        backbone = backbone.to(stored_device_back)
    if stored_device_tail != None:
        tail = tail.to(stored_device_tail)
    plt.tight_layout()
    plt.show()
    return 

def visualize_predictions(dataset_to_use, model_to_test, classes, std_dev, mean, computation_device, batch_index_to_plot=2, show_labels=True):
    data_batch = list(dataset_to_use)[batch_index_to_plot]
    model_to_test.eval()
    data, labels = data_batch
    output = model_to_test(data.to(computation_device)).detach()
    pred = output.data.max(1)[1]
    pred = pred.cpu().numpy()
    unique_labels, counts = np.unique(pred, return_counts=True)
    distribution_map = {l: count for l, count in zip(unique_labels, counts)}
    print(f'Prediction Distribution: {distribution_map}')
    fig, axs = plt.subplots(1, 6, dpi=300)
    for x in range(6):
        axs[x].imshow(unnormalize_image(data[x*24].cpu(), std_dev=std_dev, mean=mean))
        axs[x].get_xaxis().set_visible(False)
        axs[x].get_yaxis().set_visible(False)
        
        if show_labels:
            # Show label as title line 1, black color
            axs[x].set_title(f'Label: {classes[labels[x*24]]}', fontsize=5, color='black')
            
            # Add prediction text as separate text object (line 2) with color based on match
            pred_color = 'green' if pred[x*24] == labels[x*24].item() else 'red'
            axs[x].text(
                0.5, -0.1,  # position below the title, centered
                f'Prediction: {classes[pred[x*24]]}',
                fontsize=5,
                color=pred_color,
                ha='center',
                va='top',
                transform=axs[x].transAxes
            )
        else:
            # Just show prediction as title if show_labels is False
            axs[x].set_title(f'Prediction: {classes[pred[x*24]]}', fontsize=5, color='black')

    plt.tight_layout()
    plt.show()
    return

def visualize_split_predictions(dataset_to_use, head, backbone, tail, classes, std_dev, mean, computation_device, batch_index_to_plot=2, show_labels=True):    
    stored_device_head = None
    stored_device_back = None
    stored_device_tail = None
    if next(head.parameters()).device != computation_device:
        stored_device_head = next(head.parameters()).device
        head = head.to(computation_device)
    if next(backbone.parameters()).device != computation_device:
        stored_device_back = next(backbone.parameters()).device
        backbone = backbone.to(computation_device)
    if next(tail.parameters()).device != computation_device:
        stored_device_tail = next(tail.parameters()).device
        tail = tail.to(computation_device)

    data_batch = list(dataset_to_use)[batch_index_to_plot]
    head.eval()
    backbone.eval()
    tail.eval()
    data, labels = data_batch
    output = tail(backbone(head(data.to(computation_device)))).detach()
    pred = output.data.max(1)[1]
    pred = pred.cpu().numpy()
    unique_labels, counts = np.unique(pred, return_counts=True)
    distribution_map = {l: count for l, count in zip(unique_labels, counts)}
    print(f'Prediction Distribution: {distribution_map}')
    fig, axs = plt.subplots(1, 6, dpi=300)
    for x in range(6):
        axs[x].imshow(unnormalize_image(data[x*24].cpu(), std_dev=std_dev, mean=mean))
        axs[x].get_xaxis().set_visible(False)
        axs[x].get_yaxis().set_visible(False)
        
        if show_labels:
            # Show label as title line 1, black color
            axs[x].set_title(f'Label: {classes[labels[x*24]]}', fontsize=5, color='black')
            
            # Add prediction text as separate text object (line 2) with color based on match
            pred_color = 'green' if pred[x*24] == labels[x*24].item() else 'red'
            axs[x].text(
                0.5, -0.1,  # position below the title, centered
                f'Prediction: {classes[pred[x*24]]}',
                fontsize=5,
                color=pred_color,
                ha='center',
                va='top',
                transform=axs[x].transAxes
            )
        else:
            # Just show prediction as title if show_labels is False
            axs[x].set_title(f'Prediction: {classes[pred[x*24]]}', fontsize=5, color='black')

    if stored_device_head != None:
        head = head.to(stored_device_head)
    if stored_device_back != None:
        backbone = backbone.to(stored_device_back)
    if stored_device_tail != None:
        tail = tail.to(stored_device_tail)

    plt.tight_layout()
    plt.show()
    return


def save_models(path, list_model_to_save):
    if type(list_model_to_save) != list:
        list_model_to_save = [list_model_to_save]

    if os.path.exists(path):
        print(f'folder "{path}" already used')
    else:
        os.mkdir(path)
        for idx, model in enumerate(list_model_to_save):
            torch.save(model, f'{path}/model_{idx}.pt')

def load_models(path, silent =True):
    stored_model = []
    if os.path.exists(path):
        stored = os.listdir(path)
        if len(stored) > 0:
            print(f'Found {stored} models in {path}')
            for idx in range(len(stored)):
                if not silent:
                    print(f'Loading model {idx}')
                model = torch.load(f'{path}/model_{idx}.pt')
                stored_model.append(model)

        if len(stored_model) == 1:
            stored_model = stored_model[0]
            
    else:
        print(f'No stored model found in {path}')
    

    return stored_model

def evaluate_model(head, backbone, tail, test_data, poison_test_data, computation_device):
    
    ma = test_split(head, backbone, tail, test_data, computation_device)
    ba = test_split(head, backbone, tail, poison_test_data, computation_device)

    print(f'MA={ma*100:1.1f}% BA={ba*100:1.1f}%')

"""
def plot_euclidean_distance(benign_models, poisoned_models, global_model_state_dict):
    norms_of_benign_updates = [model_dist_norm(benign_model, global_model_state_dict) for benign_model in benign_models]
    norms_of_poisoned_updates = [model_dist_norm(poisoned_model, global_model_state_dict) for poisoned_model in poisoned_models]

    plt.figure(dpi=300, figsize=(6,3.5))
    plt.bar(range(1,len(benign_models) + 1), norms_of_benign_updates , color='#006400', label='Benign $L_2$-Norms')
    plt.bar(range(len(benign_models)+1, len(benign_models) + len(poisoned_models) + 1), norms_of_poisoned_updates, color='#B80F0A', label='Poisoned $L_2$-Norm')
    plt.legend(loc='best')
    plt.xticks([1,5,10,15,20,25,30])
    plt.show()

def plot_split_euclidian_distance(list_model, starting_model, malicious_index, model_split="Head"):
    norms_of_updates = []
    for i, split in enumerate(list_model):
        if i == 0:
            norms_of_updates.append(model_dist_norm(list_model[i], starting_model))
        else:
            norms_of_updates.append(model_dist_norm(list_model[i], list_model[i-1]))

    plt.figure(dpi=300, figsize=(6,3.5))
    for i in range(len(norms_of_updates)):
        if i in malicious_index:
            plt.bar(range(i+1, i + 2), norms_of_updates[i], color='#B80F0A')
        else:
            plt.bar(range(i+1, i + 2), norms_of_updates[i] , color='#006400')

    plt.title(model_split)
    plt.xticks([1,5,10,15,20,25,30])
    plt.show()
"""
def plot_euclidean_distance(list_model, malicious_index, model_split="Head"):
    norms_of_updates = []
    for i, split in enumerate(list_model):
        if i == 0:
            norms_of_updates.append(0)
        else:
            norms_of_updates.append(model_dist_norm(list_model[i], list_model[i-1]))

    plt.figure(dpi=300, figsize=(4,2))
    for i in range(len(norms_of_updates)):
        if (i) in malicious_index:
            plt.bar(range(i, i + 1), norms_of_updates[i], color='#B80F0A')
        else:
            plt.bar(range(i, i + 1), norms_of_updates[i] , color='#006400')

    plt.title(model_split)
    tick = [0,1]
    for i in range(len(list_model)):
        if (i+1) % 5 == 0:
            tick.append(i+1)
    plt.xticks(tick)
    plt.show()

"""
def plot_accepted_models(indices_of_accepted_models, benign_models, poisoned_models, global_model_state_dict):
    norms_of_all_updates = [model_dist_norm(model, global_model_state_dict) for model in tqdm(benign_models + poisoned_models)]
    norms_of_filtered_updates = [norm if norm_index in indices_of_accepted_models else 0 for norm_index, norm in enumerate(norms_of_all_updates)]
    plt.figure(dpi=300, figsize=(6,1.5))
    plt.bar(range(1,len(benign_models) + 1), norms_of_filtered_updates[:len(benign_models)], color='#006400', label='Accepted Benign $L_2$-Norms')
    plt.bar(range(len(benign_models)+1, len(benign_models) + len(poisoned_models) + 1), norms_of_filtered_updates[len(benign_models):], color='#B80F0A', label='Accepted Poisoned $L_2$-Norms')
    plt.legend(loc='best')
    plt.xticks([1,5,10,15,20,25,30])
    plt.show()
"""
def plot_accepted_models(indices_of_accepted_models, list_model, malicious_index, model_split="Head"):
    norms_of_filtered_updates = []
    last_accepted = 0
    for i, split in enumerate(list_model):
        if i == 0:
            norms_of_filtered_updates.append(0)
        else:
            if i in indices_of_accepted_models:
                norms_of_filtered_updates.append(model_dist_norm(list_model[i], list_model[last_accepted]))
                last_accepted = i
            else:
                norms_of_filtered_updates.append(0)

    plt.figure(dpi=300, figsize=(4,2))
    for i in range(len(norms_of_filtered_updates)):
        if (i) in malicious_index:
            plt.bar(range(i, i + 1), norms_of_filtered_updates[i], color='#B80F0A')
        else:
            plt.bar(range(i, i + 1), norms_of_filtered_updates[i] , color='#006400')

    plt.title(model_split)
    tick = [0,1]
    for i in range(len(list_model)):
        if (i+1) % 5 == 0:
            tick.append(i+1)
    plt.xticks(tick)
    plt.show()


def solution(key=""):
    to_find = get_hash(key, 2000)
    for task in range(1, len(HASHES)):
        if to_find == HASHES[task]:
            display(Markdown(dencrypt_string(SOLUTIONS[task] ,key)))
            return
    display(Markdown(dencrypt_string(SOLUTIONS[0] , "You Should Focus On The Task")))

def encrypt_string(string, key):
    seed = lambda s: int(s, 16) % MAX_CHR
    encrypted = ""
    for char in string:
        key = get_hash(key)
        encrypted += chr((ord(char) + seed(key)) % MAX_CHR)

    return encrypted

def dencrypt_string(string, key):
    seed = lambda s: int(s, 16) % MAX_CHR
    decrypted = ""
    for char in string:
        key = get_hash(key)
        decrypted += chr((ord(char) - seed(key)) % MAX_CHR)
    return decrypted

def get_hash(key, repeat=1):
    hash_value = key
    while repeat > 0:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(hash_value.encode('utf-8'))
        hash_value = sha256_hash.hexdigest()
        repeat -= 1
    return hash_value


