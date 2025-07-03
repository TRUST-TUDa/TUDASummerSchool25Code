def calculate_hash_of_tensor(data):
    """
    Calculates were primitive hash value of a tensor.
    This hash is not secure!
    It is only sufficient to notice accidental changes at the tensors (to notice anoying bugs)
    """
    return float(data.double().mean())

def get_model_hash(model_to_hash):
    """
    Insecure hash function of a model's parameters
    """
    hashs_of_model = {}
    for name, data in model_to_hash.items():
        hashs_of_model[name] = calculate_hash_of_tensor(data)
    return hashs_of_model

def check_hashs(models, hash_values):
    for model, hashs_of_model in zip(models, hash_values):
        for name, data in model.items():
            hash_value = calculate_hash_of_tensor(data)
            if hash_value != hashs_of_model[name]:
                raise Exception('Your implementation has changed the benign models.\nThis causes unexpected behavior. Please check you implementation and __rerain all benign models__')
    return 

def check_hashs_single_model(model, hashs_of_model):
    for name, data in model.items():
        hash_value = calculate_hash_of_tensor(data)
        if hash_value != hashs_of_model[name]:
            raise Exception('Your implementation has changed the model.\nThis causes unexpected behavior. Please check you implementation and __retrain all benign models__')
    return 

def get_models_hash(models):
    hash_values = []
    
    for model in models:
        hashs_of_model = {}
        for name, data in model.items():
            hashs_of_model[name] = calculate_hash_of_tensor(data)
        hash_values.append(hashs_of_model)
    return hash_values