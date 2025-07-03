import numpy as np
import torch
from TUDASummerSchool25Code.ModelStateDictNames import NAMES_OF_AGGREGATED_PARAMETERS


def test(model, dataloader, computation_device):
    
    predictions = []
    correct = []
    loss_values = []
    model.eval()
    total_samples = 0
    for batch_id, batch in enumerate(dataloader):
        data, targets = batch
        total_samples += targets.shape[0]
        output = model(data.to(computation_device)).detach()
        targets = targets.to(computation_device)
        predictions.append(output)
        correct.append(targets)
    model.train()
    correct = torch.cat(correct)
    predictions = torch.cat(predictions)
    predictions = torch.argmax(predictions, dim=1)
    correct = torch.eq(predictions, correct).sum()
    return (correct/total_samples).cpu().item()

def test_split(head, backbone, tail, dataloader, computation_device):
    
    predictions = []
    correct = []
    loss_values = []
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
    head.eval()
    backbone.eval()
    tail.eval()
    total_samples = 0
    for batch_id, batch in enumerate(dataloader):
        data, targets = batch
        total_samples += targets.shape[0]
        output = tail(backbone(head(data.to(computation_device)))).detach()
        targets = targets.to(computation_device)
        predictions.append(output)
        correct.append(targets)
    head.train()
    backbone.train()
    tail.train()
    correct = torch.cat(correct)
    predictions = torch.cat(predictions)
    predictions = torch.argmax(predictions, dim=1)
    correct = torch.eq(predictions, correct).sum()
    if stored_device_head != None:
        head = head.to(stored_device_head)
    if stored_device_back != None:
        backbone = backbone.to(stored_device_back)
    if stored_device_tail != None:
        tail = tail.to(stored_device_tail)

    return (correct/total_samples).cpu().item()

def do_save_division(dividend, divisor, zero_value='-'):
    if divisor == 0:
        return zero_value
    return dividend / divisor

def extract_weights(local_model, to_cpu=True):
    """
    clones weights
    """
    result = {}
    if isinstance(local_model, dict):
        items = local_model.items()
    else:
        items = local_model.state_dict().items()

    for layer_name, local_layer in items:
        if to_cpu:
            local_layer = local_layer.cpu()
        result[layer_name] = local_layer.detach().clone()
    return result

def model_dist_norm(model1, model2):
    squared_sum = 0
    for name, layer in model1.items():
        if name not in NAMES_OF_AGGREGATED_PARAMETERS:
            continue
        squared_sum += torch.sum(torch.pow(layer.data.cpu() - model2[name].data.cpu(), 2))
    return torch.sqrt(squared_sum)


def model_dist_norm_var(model, target_params):
    """
    flexible function for determining a norm of a model without losing computation graph
    """
    #assert not isinstance(model, dict)
    squared_sum = None
    is_first_layer = True
    for name, layer in model.named_parameters():
        if name not in NAMES_OF_AGGREGATED_PARAMETERS:
            continue
        sum_of_current_layer = torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        if is_first_layer:
            squared_sum = sum_of_current_layer
        squared_sum += sum_of_current_layer
    assert squared_sum is not None
    return torch.sqrt(squared_sum)

"""
def evaluate_model_filtering(indices_of_accepted_models, number_of_adversaries, number_of_benign_clients):
    indices_of_accepted_models = np.array(indices_of_accepted_models)
    tn = np.where(indices_of_accepted_models < number_of_benign_clients)[0].shape[0]
    assert 0 <= tn <= number_of_benign_clients
    fn = np.where(indices_of_accepted_models >= number_of_benign_clients)[0].shape[0]
    assert 0 <= fn <= number_of_adversaries, f'FN={fn}, number_of_adversaries={number_of_adversaries}, number_of_benign_clients={number_of_benign_clients}, indices_of_accepted_models={indices_of_accepted_models}'
    tp = number_of_adversaries - fn
    assert 0 <= tp <= number_of_adversaries
    fp = number_of_benign_clients - tn
    assert 0 <= fp <= number_of_benign_clients
    tnr = tn/number_of_benign_clients
    assert 0 <= tnr <= 1
    tpr = do_save_division(tp, number_of_adversaries)
    assert 0 <= tpr <= 1
    precision = do_save_division(tp, tp + fp)
    assert 0 <= precision <= 1
    f1_score = do_save_division(2 * tp, 2 * tp + fp + fn)
    assert 0 <= f1_score <= 1
    print(f'TNR = {tnr*100:1.2f}%')
    print(f'TPR = {tp/number_of_adversaries*100:1.2f}% (Recall)')
    print(f'Precision = {precision*100:1.2f}%')
    print(f'F1-Score = {f1_score:1.2f}')
"""

def evaluate_model_filtering(indices_of_accepted_models, malicious_index, benign_index):
    indices_of_accepted_models = np.array(indices_of_accepted_models)
    tn = np.sum([1 if i in indices_of_accepted_models else 0 for i in benign_index])
    assert 0 <= tn <= len(benign_index)
    fn = np.sum([1 if i in indices_of_accepted_models else 0 for i in malicious_index])
    assert 0 <= fn <= len(malicious_index), f'FN={fn}, number_of_adversaries={len(malicious_index)}, number_of_benign_clients={len(benign_index)}, indices_of_accepted_models={indices_of_accepted_models}'

    tp = len(malicious_index) - fn
    assert 0 <= tp <= len(malicious_index)
    fp = len(benign_index) - tn
    assert 0 <= fp <= len(benign_index)

    tnr = tn/len(benign_index)
    assert 0 <= tnr <= 1

    tpr = do_save_division(tp, len(malicious_index))
    assert 0 <= tpr <= 1

    precision = do_save_division(tp, tp + fp)
    assert 0 <= precision <= 1
    f1_score = do_save_division(2 * tp, 2 * tp + fp + fn)
    assert 0 <= f1_score <= 1
    print(f'TNR = {tnr*100:1.2f}%')
    print(f'TPR = {tp/len(malicious_index)*100:1.2f}% (Recall)')
    print(f'Precision = {precision*100:1.2f}%')
    print(f'F1-Score = {f1_score:1.2f}')

def scale_update_from_model(model, target_model_params, scaling_factor):
    """
    Scales all parameters of a model update U, for a given model m=U+g, where g is the global model
    (here the target model)
    """
    local_weights = {}
    if not isinstance(model, dict):
        model = model.state_dict()
    for name, data in model.items():
        if name not in NAMES_OF_AGGREGATED_PARAMETERS:
            local_weights[name] = data
            continue
        target_value = target_model_params[name]
        new_value = target_value + (data - target_value) * scaling_factor
        local_weights[name] = new_value

    return local_weights

def get_one_vec_sorted_layers(model, layer_names, size=None):
    """
    Converts a model, given as dictionary type, to a single vector
    """
    if size is None:
        size = 0
        for name in layer_names:
            size += model[name].view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    size = 0
    for name in layer_names:
        layer_as_vector = model[name].view(-1)
        layer_width = layer_as_vector.shape[0]
        sum_var[size:size + layer_width] = layer_as_vector
        size += layer_width
    return sum_var

def euclidean_distance(tensor1, tensor2):
    """
    This function calculates the Euclidian Distance between 2 vectors (tensors)
    :param tensor1: first tensor
    :param tensor2: second tensor
    :return floating number, representing the Euclidean distance between both given vectors (tensors)
    """
    squared_differences = (tensor1 - tensor2)**2
    summed_squared_differences = torch.sum(squared_differences)
    euclidean_dist = torch.sqrt(summed_squared_differences)

    return euclidean_dist

def pairwise_euclidian_distance(models):
    """
    This function calculates the Euclidian Distance between each model of the federation
    :param models: list of all local models' state dict
    :return list, containing for each model i, its Euclidian Distance to model j
    """

    # Create a matrix that will contain each pairwise distance
    all_distances = np.zeros((len(models), len(models))) 

    models_as_vector = []
    for model in models:
        models_as_vector.append(get_one_vec_sorted_layers(model, NAMES_OF_AGGREGATED_PARAMETERS))

    for i in range(len(models_as_vector)):
        for j in range(i + 1, len(models_as_vector)):
            distance = euclidean_distance(models_as_vector[i], models_as_vector[j])
            all_distances[i, j] = distance
            all_distances[j, i] = distance

    assert np.count_nonzero(np.diagonal(all_distances)) == 0, f'The distance of model I from itself (model I) should be zero'
    
    return all_distances

def load_split_dict(model_dict, split_model, map_location="cpu"):
    """
    Splitting and Loading of a centralized model dict into the submodel ( client or server )
    :param model_dict: dictionary of the original unsplit model
    :param split_model: split section of the model ( client or server )
    """

    new_dict = {}
    for split_key in split_model.state_dict().keys():
        if split_key in model_dict.keys():        
            if split_key in NAMES_OF_AGGREGATED_PARAMETERS:
                new_dict[split_key] = model_dict[split_key].to(map_location)
    assert len(new_dict.keys()) > 0, f'You did not create a new dictionary'
    split_model = split_model.to(map_location)
    split_model.load_state_dict(new_dict)