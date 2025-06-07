
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


@st.cache_data
def train_model():
    
    df = pd.read_csv('data.csv', encoding='utf-8')
    

    X = df['Metin']
    y = df['Etiket']
    

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X.astype(str))
    
    model = MultinomialNB()
    model.fit(X_vectorized, y)
    

    return model, vectorizer


model, vectorizer = train_model()




st.title('🎣 Phishwall - Oltalama Tespit Aracı')


st.write('Analiz etmek istediğiniz şüpheli e-posta metnini aşağıdaki kutucuğa yapıştırın ve butona tıklayın.')


user_input = st.text_area('E-posta Metni', height=200)


if st.button('Analiz Et'):
    if user_input:
      
        test_vector = vectorizer.transform([user_input])
        
     
        prediction = model.predict(test_vector)
        prediction_proba = model.predict_proba(test_vector)

       
        st.subheader('Analiz Sonucu')

        if prediction[0] == 'Tehlikeli':
            tehlike_yuzdesi = prediction_proba[0][1] * 100
            st.error(f'Bu metin %{tehlike_yuzdesi:.2f} olasılıkla TEHLİKELİDİR!')
            st.warning('Bu metindeki linklere tıklamamanız veya herhangi bir bilgi paylaşmamanız önerilir.')
        else:
            guvenli_yuzdesi = prediction_proba[0][0] * 100
            st.success(f'Bu metin %{guvenli_yuzdesi:.2f} olasılıkla GÜVENLİDİR.')
    else:
        st.warning('Lütfen analiz etmek için bir metin girin.')
