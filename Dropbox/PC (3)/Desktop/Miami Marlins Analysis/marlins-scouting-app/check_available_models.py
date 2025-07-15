import pickle

def check_available_models():
    """Check what models are available in the sequential_models.pkl file"""
    try:
        with open('sequential_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        print("Available models in sequential_models.pkl:")
        print("=" * 50)
        for key in models.keys():
            print(f"  - {key}")
            
        print(f"\nTotal models: {len(models)}")
        
    except Exception as e:
        print(f"Error loading models: {e}")

if __name__ == "__main__":
    check_available_models() 
 
 
 
 
 
 
 
 
 
 
 
 
 
 