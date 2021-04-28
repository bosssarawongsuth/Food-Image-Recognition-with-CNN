import tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import pandas as pd



header_image = Image.open('./icon.png')
###STYLE###

st.set_page_config(page_title="Food Image Classification⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",page_icon=header_image)


hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


st.markdown("""
<style>
.big-font {
    font-size:75px !important;
}
</style>

<style>
.title-style {
    text-align: left;
    vertical-align: middle;

}
</style>
""", unsafe_allow_html=True)



##########

model = tensorflow.keras.models.load_model('best_model.hdf5')

foodinfo = pd.read_csv("./foodinfo.csv")


food_dict = {0: 'Bibimbap',
 1: 'Caesar Salad',
 2: 'Carrot Cake',
 3: 'Chicken Curry',
 4: 'Club Sandwich',
 5: 'Edamame',
 6: 'Fish and Chips',
 7: 'French Fries',
 8: 'French Toast',
 9: 'Fresh Spring Roll',
 10: 'Fried Calamari',
 11: 'Fried Rice',
 12: 'Grilled Salmon',
 13: 'Gyoza',
 14: 'Hot and Sour Soup',
 15: 'Hummus',
 16: 'Miso Soup',
 17: 'Omelette',
 18: 'Pad Thai',
 19: 'Peking Duck',
 20: 'Pho',
 21: 'Ramen',
 22: 'Samosa',
 23: 'Sashimi',
 24: 'Seaweed Salad',
 25: 'Spring Rolls',
 26: 'Steamed Mussels',
 27: 'Sushi',
 28: 'Xiao Long Bao'}



def import_and_predict(image_data, model):

	size = (299,299)    
	image = ImageOps.fit(image_data, size, Image.ANTIALIAS)

    #my_image = load_img('./datasets/food-101/images/club_sandwich/2415.jpg', target_size=(299, 299))

	#preprocess the image
	my_image = img_to_array(image)
	my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
	my_image = preprocess_input(my_image)

	


	prediction = model.predict(my_image)

	
	return prediction



#### PAGE CONTENT ####
col1, col2 = st.beta_columns((1,5))
col1.image(header_image,use_column_width=True)
col2.markdown("<h1 class='title-style'>Food Image Classification</h1>", unsafe_allow_html=True)



st.write("""
		 ## DSi Capstone Project
		 """
		 )

st.write("Jetnipat Sarawongsuth (Boss)")
file = st.file_uploader("Upload your food image below", type=["jpg", "png"])


if file is None:
	#st.subheader("Please upload an image file")
	pass
else:

	image = Image.open(file)
	prediction = import_and_predict(image, model)
	argmax = np.argmax(prediction)
	#st.subheader(f"{food_dict[argmax]} ({str(prediction[0][argmax])[:4]})")

	st.write(f"""
		 ### **{food_dict[argmax]}** ({str(prediction[0][argmax])[:4]})
		 """
		 )
	st.image(image, use_column_width=True)
	

	
	#st.markdown(f'<p class="big-font">{food_dict[argmax]} with {prediction[0][argmax]} !!</p>', unsafe_allow_html=True)

	prediction = prediction[0]

	keys = np.array(list(food_dict.values()))


	pred = pd.DataFrame({"Food":keys,"Probability":prediction}).sort_values(by=["Probability"],ascending=False)


	col3, col4 = st.beta_columns(2)

	col3.subheader("Classification Breakdown: ")
	col3.dataframe(pred.head().set_index("Food"))


	food_fact = (foodinfo.loc[foodinfo['Food'] == food_dict[argmax]]).T
	food_fact.columns = ['']
	col4.subheader("Nutrition Facts (per 100 grams):")
	col4.dataframe(food_fact)





