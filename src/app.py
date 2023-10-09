import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import cv2
import time



# Define a function to load the pickle file
def load_pickle(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)

# Call the function to load your pickle file
data_dict = load_pickle('src\streamlit_toolkit.pkl')

# Now, you can access the components you saved in the dictionary where .
scaler = data_dict.get("scaler")
model1 = data_dict.get("model1")
#model2 = data_dict.get("model2")

def main():

    # Setup web page
    st.set_page_config(

     page_title="Favorita Sales Prediction App",
     #page_icon=('nav_pg.jpg'),
     layout="wide",
     menu_items={
         'About': "The source code for this application can be accessed on GitHub https://github.com/florenceaffoh/fund.-of-streamlit.git"
     }
)
    st.markdown("""
    <style type="text/css">
    blockquote {
        margin: 1em 0px 1em -1px;
        padding: 0px 0px 0px 1.2em;
        font-size: 20px;
        border-left: 5px solid rgb(230, 234, 241);
        # background-color: rgb(129, 164, 182);
    }
    blockquote p {
        font-size: 30px;
        color: #FFFFFF;
    }
    [data-testid=stSidebar] {
        background-color: rgb(129, 164, 182);
        color: #FFFFFF;
    }
    [aria-selected="true"] {
         color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

    st.sidebar.title("**Favorita Sales Prediction App**")
    st.sidebar.write('## Navigation')
    st.sidebar.markdown("Made by TEAM CHARLESTON")
    page = st.sidebar.radio('Select a Page', ['Home', 'Predict', 'Explore', 'About'])
    if page == 'Home':
         home()
    elif page == 'Predict':
        predict_sales()
    elif page == 'Explore':
        explore_data()
    elif page == 'About':
        about_info()

def home():

        img = cv2.imread(r"C:\Users\HP\OneDrive - Azubi Africa\Pictures\sales.jpg")
    # Rotate the image 90 degrees clockwise
        #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        st.image(img, channels='RGB')


        st.title("Welcome to Favorita Sales Prediction App")
        st.write("""
### To ensure the best of **Experience** on the Favorita Sales Prediction App,
Navigate using the sidebar(the arrow in the top left corner) to:
1. Predict sales
2. Explore the dataset
3. Learn more about this project.
""")

def predict_sales():
    st.title("Predict Sales")
    
    st.write("This is used to predict sales per stores according to various factors")

    # Form for prediction
    with st.form(key="prediction_form"):
        id = st.number_input("put index number:")
        family =st.selectbox("What Family is your product categorize under:",['AUTOMOTIVE', 'BABY CARE','BEAUTY', 'BEVERAGES', 'BOOKS','BREAD/BAKERY', 'CELEBRATION', 'CLEANING',
                                                                              'DAIRY', 'family_DELI', 'family_EGGS', 'family_FROZEN FOODS','GROCERY I', 'GROCERY II', 'HARDWARE',
                                                                              'HOME AND KITCHEN I', 'HOME AND KITCHEN II','HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR','LAWN AND GARDEN',
                                                                              'LINGERIE', 'LIQUOR,WINE,BEER','MAGAZINES', 'MEATS', 'PERSONAL CARE','PET SUPPLIES', 'PLAYERS AND ELECTRONICS',
                                                                              'POULTRY', 'PREPARED FOODS', 'PRODUCE','SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'])
        store_city =st.selectbox("What city does the product under goes:",['Ambato', 'Babahoyo', 'Cayambe','Cuenca', 'Daule', 'El Carmen',
                                                                           'Esmeraldas', 'Guaranda', 'Guayaquil','Ibarra', 'Latacunga', 'Libertad',
                                                                           'Loja', 'Machala', 'Manta','Playas', 'Puyo', 'Quevedo','Quito', 'Riobamba',
                                                                           'Salinas','Santo Domingo'])
        oil_price = st.number_input("The Price of the oil:")
        store_nbr =st.slider("What store number of the product:", 1,54)
        holiday = st.selectbox("Is Today a Holiday :", ['Holiday', 'Not Holiday', 'Work Day', 'Additional', 'Event', 'Transfer','Bridge'])
        locale= st.radio("What holiday category does it fall under :", ['National', 'Not Holiday', 'Local', 'Regional'])
        Transferred = st.radio("Is the Holiday Transferred :", ['Yes', 'No'])
        onpromotion = st.number_input("What items on Promotion : ")
        
        # Submission of the form
        if st.form_submit_button("Predict"):
            try:
                df_input = pd.DataFrame({
                    "id":[id],
                    "family":[family],
                    "oil_price":[oil_price],
                    "store_nbr":[store_nbr],
                    "store_city":[store_city],
                    "holiday": [holiday],
                    "locale": [locale],
                    "Transferred": [Transferred],
                    "onpromotion": [onpromotion]
                })

                # One-hot encode the data
                df_encoded = pd.get_dummies(df_input)

                # Ensuring the input data has the same columns as training data
                # Here, 'model_columns' is a list of columns used in the trained model.
                # This might come from the training data's columns after one-hot encoding.
                model_columns = ['id', 'store_nbr', 'onpromotion', 'oil_price','AUTOMOTIVE', 'CARE','BEAUTY', 'BEVERAGES', 'BOOKS','BREAD/BAKERY', 'CELEBRATION',
                                 'CLEANING','DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS','GROCERY I', 'GROCERY II', 'HARDWARE','HOME AND KITCHEN I', 'HOME AND KITCHEN II',
                                 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR','LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER','MAGAZINES', 'MEATS', 'PERSONAL CARE',
                                 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS','POULTRY', 'PREPARED FOODS', 'PRODUCE','SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD','Ambato',
                                 'Babahoyo', 'Cayambe','Cuenca', 'Daule', 'El Carmen','Esmeraldas', 'Guaranda', 'Guayaquil','Ibarra', 'Latacunga', 'Libertad','Loja',
                                 'Machala', 'Manta','Playas', 'Puyo', 'Quevedo','Quito', 'Riobamba', 'Salinas','Santo Domingo', 'Holiday','No Event']

                for col in model_columns:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0

                # Reorder columns based on model
                df_encoded = df_encoded[model_columns]

                # Now you can predict
                predictions = model1.predict(df_encoded)

                # Display predicted sales
                st.balloons()
                st.progress(10)
                with st.spinner('Wait for it...'):    
                    time.sleep(10)

                st.success(f"The Predicted Sales is: ${predictions[0]}", icon="✅")

            except:
                st.error(f"Something went wrong during the Sales Prediction", icon="🚨")

def explore_data():
    st.title("Explore Data")
    df = pd.read_csv('sales.csv')
    st.write(df)

def about_info():
    #st.image("nav_pg.jpg")
    st.title("About")
    st.write("""## FAVORITA SALES PREDICTION APP
             
Beyond the bustling aisles and checkout lanes of grocery stores, there lies a complex web of purchasing patterns,
seasonal trends, and intricate sales dynamics. Understanding and predicting these dynamics can be the key to a retailer's success or downfall.
For Corporation Favorita, one of Ecuador's largest grocery retailers, these patterns translate to thousands of items across multiple store locations.

With the rise of data science and advanced analytical techniques,time series forecasting has emerged as an invaluable tool for businesses like Favorita.
We embarked on a journey to harness the power of time series forecasting,
aiming to craft a robust predictive model that can more accurately 
forecast the unit sales of Favorita's vast product range across its numerous stores. 
             
This app is a simple Streamlit application interface to a regression analysis done on The company Favorita sales data to predict sales.
Developed by Team Charleston during Azubi Cohort 4.
More details about the project and methodology can be found [here](https://github.com/florenceaffoh/LP3-TimeSeriesAnalysis.git).
""")

if __name__ == '__main__':
    main()