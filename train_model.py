import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def fix_and_save_model(old_model_path='model_isl.h5', new_model_path='model_isl_fixed.h5'):
    """
    Purane Keras model ko load karke naye format mein save karta hai taaki 'batch_shape' error na aaye.
    """
    
    # Check karein ki purani model file maujood hai ya nahi
    if not os.path.exists(old_model_path):
        print(f"Error: Purani model file '{old_model_path}' nahi mili.")
        print("Kripya is script ko model file ke saath ek hi folder mein rakhein.")
        return

    print(f"Purani model file '{old_model_path}' se weights load kiye jaa rahe hain...")

    # 1. Model ka architecture (blueprint) define karein.
    # Yeh aapke purane model 'model_isl.h5' ke structure se match karta hai.
    # Input 42 features ka hai (ek haath ke liye).
    model = keras.Sequential([
        layers.Input(shape=(42,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(35, activation='softmax') # Aapke model mein 35 classes thi
    ])
    
    print("\nModel ka naya structure taiyaar hai:")
    model.summary()

    try:
        # 2. Sirf weights ko purane model se load karein.
        model.load_weights(old_model_path)
        print(f"\nWeights safaltapoorvak '{old_model_path}' se load ho gaye hain.")
        
        # 3. Naye, theek kiye hue model ko save karein.
        # Yeh file bina kisi error ke load ho jaayegi.
        model.save(new_model_path)
        print(f"Clean model safaltapoorvak '{new_model_path}' naam se save ho gaya hai.")
        
    except Exception as e:
        print(f"\nEk error aa gaya: {e}")
        print("Ho sakta hai ki model ka architecture match na kar raha ho.")

if __name__ == '__main__':
    fix_and_save_model()
