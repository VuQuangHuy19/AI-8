from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model

def create_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = Flatten()(base_model.output)
    
    # Nhánh dự đoán tuổi
    age_output = Dense(1, activation='linear', name='age_output')(x)
    
    # Nhánh dự đoán giới tính
    gender_output = Dense(2, activation='softmax', name='gender_output')(x)
    
    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])
    model.compile(
        optimizer='adam',
        loss={'age_output': 'mae', 'gender_output': 'categorical_crossentropy'},
        metrics={'age_output': 'mae', 'gender_output': 'accuracy'}
    )
    return model
