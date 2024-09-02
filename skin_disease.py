

import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

# Paths to images
image_paths = {
    "Home Page": "Picture1.png",
    "Prediction Page": "Picture2.png",
    "About Models Page": "Picture3.png",
    "More About Disease Page": "Picture4.png",
}

# Display an image for the given page
def display_image(page_name):
    st.image(image_paths.get(page_name, "Picture2.png"), use_column_width=True)

# Custom focal loss function
def focal_loss_fixed(gamma=2., alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(fl)
    return focal_loss

# Register custom loss function
get_custom_objects()['focal_loss_fixed'] = focal_loss_fixed

# Load models
def load_models():
    try:
        models = {
            'EfficientNetB0': load_model('EfficientNetB0_disease_model.keras', custom_objects={'focal_loss_fixed': focal_loss_fixed}),
            'DenseNet121': load_model('DenseNet121_disease_model.keras', custom_objects={'focal_loss_fixed': focal_loss_fixed})
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

models = load_models()

# Disease categories
disease_categories = [
    'Acne and Rosacea', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis', 'Bullous Disease', 'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia and other Hair Diseases',
    'Herpes HPV and other STDs', 'Light Diseases and Disorders of Pigmentation',
    'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles',
    'Nail Fungus and other Nail Disease', 'Poison Ivy and other Contact Dermatitis',
    'Psoriasis Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites',
    'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives',
    'Vascular Tumors', 'Vasculitis', 'Warts Molluscum and other Viral Infections'
]

# Image prediction
def predict_image(model, image):
    try:
        img = image.resize((224, 224))  # Resize to model input size
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        return disease_categories[predicted_class]
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return "Error in prediction"

# Page functions
def home_page():
    display_image("Home Page")
    st.title('Skin Disease Prediction App')
    st.write('Welcome to the Skin Disease Prediction App. Use the navigation menu to access different features of the app.')

def prediction_page():
    display_image("Prediction Page")
    st.title('Prediction Page')
    st.write('Upload an image of the skin and choose a model to get predictions.')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    model_name = st.selectbox("Choose a model", list(models.keys()))

    if uploaded_file and model_name:
        image = Image.open(uploaded_file)
        model = models.get(model_name)
        if model:
            prediction = predict_image(model, image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write(f'Prediction: {prediction}')
        else:
            st.error("Model not found.")

def about_models_page():
    display_image("About Models Page")
    st.title('About Models')
    st.write('Information about the models used for predictions:')
    st.write('* **EfficientNetB0**: A highly efficient and lightweight convolutional neural network (CNN) model that balances accuracy and computational cost by scaling depth, width, and resolution uniformly.')
    st.write('* **DenseNet121**: Known for its dense connectivity, where each layer receives inputs from all preceding layers, leading to improved feature propagation, efficient parameter usage, and enhanced accuracy.')
    st.write('* **InceptionV3**: A model known for its use of multiple filter sizes in a single layer, allowing it to capture diverse features at different scales. It’s efficient and often used for various image classification tasks.')
    st.write('* **ResNet152V2**: A deeper version of the Residual Networks (ResNet) architecture, with 152 layers, using skip connections (or shortcuts) to solve the vanishing gradient problem in deep networks, leading to improved accuracy and stability in training.')

def more_about_disease_page():
    st.container()
    st.image('Picture4.png', use_column_width=True)
    st.title('More About Diseases')
    st.write('Detailed information about various skin diseases:')

    st.write('- **Acne and Rosacea**: Conditions affecting the skin with red pimples, inflammation, and sometimes scarring. Rosacea can cause redness and visible blood vessels on the face.')
    st.write('- **Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions**: Precancerous and cancerous lesions caused by sun exposure, often appearing as rough, scaly patches or nodules.')
    st.write('- **Atopic Dermatitis**: A chronic skin condition characterized by dry, itchy, and inflamed skin. Often associated with allergies and asthma.')
    st.write('- **Bullous Disease**: A group of conditions causing large, fluid-filled blisters on the skin, often resulting from autoimmune disorders.')
    st.write('- **Cellulitis Impetigo and other Bacterial Infections**: Bacterial infections that can cause red, swollen, and painful areas on the skin. Impetigo is a contagious infection that causes red sores.')
    st.write('- **Eczema**: A condition where patches of skin become inflamed, itchy, cracked, and rough. Blisters may sometimes occur.')
    st.write('- **Exanthems and Drug Eruptions**: Widespread rashes often caused by infections or allergic reactions to medications.')
    st.write('- **Hair Loss Alopecia and other Hair Diseases**: Conditions leading to hair thinning or baldness, including autoimmune-related hair loss (alopecia) and other disorders affecting hair growth.')
    st.write('- **Herpes HPV and other STDs**: Viral infections such as herpes simplex virus (HSV) causing sores, and human papillomavirus (HPV) causing warts. These are commonly transmitted through sexual contact.')
    st.write('- **Light Diseases and Disorders of Pigmentation**: Conditions like vitiligo that cause loss of skin color, and other disorders affecting pigmentation, including sensitivity to sunlight.')
    st.write('- **Lupus and other Connective Tissue Diseases**: Autoimmune diseases that cause inflammation and damage to various body tissues, including the skin, often resulting in rashes and other skin issues.')
    st.write('- **Melanoma Skin Cancer Nevi and Moles**: Melanoma is a serious form of skin cancer that develops in melanocytes. Moles are common skin growths that are usually harmless but can sometimes become cancerous.')
    st.write('- **Nail Fungus and other Nail Disease**: Infections and conditions affecting the nails, causing discoloration, thickening, and sometimes pain. Fungal infections are common in toenails.')
    st.write('- **Poison Ivy and other Contact Dermatitis**: Skin reactions caused by contact with irritants or allergens, leading to redness, itching, and blistering. Poison ivy causes an allergic reaction from contact with its sap.')
    st.write('- **Psoriasis Lichen Planus and related diseases**: Psoriasis is a chronic autoimmune condition that causes rapid skin cell turnover, resulting in scaling and inflammation. Lichen planus causes purplish, itchy, flat-topped bumps.')
    st.write('- **Scabies Lyme Disease and other Infestations and Bites**: Infestations like scabies are caused by mites that burrow into the skin, causing intense itching. Lyme disease is transmitted through tick bites and can cause skin rashes and other symptoms.')
    st.write('- **Seborrheic Keratoses and other Benign Tumors**: Non-cancerous growths on the skin that are usually brown, black, or light tan. They can appear on any part of the body and are generally harmless.')
    st.write('- **Systemic Disease**: Conditions that affect multiple organs and tissues, including the skin. Skin manifestations can be a sign of underlying systemic diseases.')
    st.write('- **Tinea Ringworm Candidiasis and other Fungal Infections**: Fungal infections of the skin, hair, and nails, such as ringworm (tinea) and candidiasis, often cause redness, itching, and scaling.')
    st.write('- **Urticaria Hives**: A skin reaction that causes red, itchy welts. Hives can be triggered by allergies, stress, or other factors.')
    st.write('- **Vascular Tumors**: Growths that develop from blood vessels, including hemangiomas and angiosarcomas. These can range from benign to malignant.')
    st.write('- **Vasculitis**: Inflammation of blood vessels that can cause changes in the skin, such as red or purple spots, and can affect various organs.')
    st.write('- **Warts Molluscum and other Viral Infections**: Warts are small, rough growths caused by HPV, while molluscum contagiosum causes small, painless bumps on the skin due to a poxvirus infection.')


# Page navigation
PAGES = {
    "Home Page": home_page,
    "Prediction Page": prediction_page,
    "About Models Page": about_models_page,
    "More About Disease Page": more_about_disease_page
}

# Main function
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    with st.spinner(f"Loading {selection} ..."):
        page()

if __name__ == "__main__":
    main()
