import pickle

# Load the models file
try:
    with open('sequential_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    print("Models loaded successfully!")
    print(f"Type: {type(models)}")
    print(f"Keys: {list(models.keys())}")
    
    # Print details for each key
    for key, value in models.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if hasattr(value, '__dict__'):
            print(f"  Attributes: {list(value.__dict__.keys())}")
        elif isinstance(value, dict):
            print(f"  Dict keys: {list(value.keys())}")
        else:
            print(f"  Value: {value}")
            
except Exception as e:
    print(f"Error loading models: {e}") 

# Load the models file
try:
    with open('sequential_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    print("Models loaded successfully!")
    print(f"Type: {type(models)}")
    print(f"Keys: {list(models.keys())}")
    
    # Print details for each key
    for key, value in models.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if hasattr(value, '__dict__'):
            print(f"  Attributes: {list(value.__dict__.keys())}")
        elif isinstance(value, dict):
            print(f"  Dict keys: {list(value.keys())}")
        else:
            print(f"  Value: {value}")
            
except Exception as e:
    print(f"Error loading models: {e}") 
 

# Load the models file
try:
    with open('sequential_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    print("Models loaded successfully!")
    print(f"Type: {type(models)}")
    print(f"Keys: {list(models.keys())}")
    
    # Print details for each key
    for key, value in models.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if hasattr(value, '__dict__'):
            print(f"  Attributes: {list(value.__dict__.keys())}")
        elif isinstance(value, dict):
            print(f"  Dict keys: {list(value.keys())}")
        else:
            print(f"  Value: {value}")
            
except Exception as e:
    print(f"Error loading models: {e}") 

# Load the models file
try:
    with open('sequential_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    print("Models loaded successfully!")
    print(f"Type: {type(models)}")
    print(f"Keys: {list(models.keys())}")
    
    # Print details for each key
    for key, value in models.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if hasattr(value, '__dict__'):
            print(f"  Attributes: {list(value.__dict__.keys())}")
        elif isinstance(value, dict):
            print(f"  Dict keys: {list(value.keys())}")
        else:
            print(f"  Value: {value}")
            
except Exception as e:
    print(f"Error loading models: {e}") 
 
 
 
 
 
 