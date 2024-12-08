import streamlit as st



user_input = st.chat_input()

age = st.slider("How old are you?", 0, 130, 25, key='age_slider')
st.write("I'm ", st.session_state['age_slider'], "years old")
