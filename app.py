# Gerekli kütüphaneleri içeri aktarıyoruz
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#----------------------------------------------------------------
# YAPAY ZEKA MODELİNİ EĞİTME BÖLÜMÜ
# Bu bölüm, Streamlit'in @st.cache_data özelliği sayesinde sadece bir kez çalışır.
# Bu sayede uygulama her kullanıldığında modeli tekrar tekrar eğitmez, çok hızlı çalışır.
#----------------------------------------------------------------
@st.cache_data
def train_model():
    # CSV dosyasını okuyoruz. Hata olmaması için encoding belirtiyoruz.
    df = pd.read_csv('data.csv', encoding='utf-8')
    
    # Gerekli sütunları alıyoruz
    X = df['Metin']
    y = df['Etiket']
    
    # Metin verilerini sayısala dönüştürüyoruz
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X.astype(str))
    
    # Modeli oluşturup eğitiyoruz
    model = MultinomialNB()
    model.fit(X_vectorized, y)
    
    # Eğitilmiş modeli ve vektörleştiriciyi geri döndürüyoruz
    return model, vectorizer

# Modeli ve vektörleştiriciyi yüklüyoruz
model, vectorizer = train_model()

#----------------------------------------------------------------
# WEB UYGULAMASININ ARAYÜZÜNÜ OLUŞTURMA BÖLÜMÜ
# Streamlit komutları ile web sayfasını tasarlıyoruz.
#----------------------------------------------------------------

# Sayfa başlığını yazdırıyoruz
st.title('🎣 Phishwall - Oltalama Tespit Aracı')

# Kullanıcıya bilgi veriyoruz
st.write('Analiz etmek istediğiniz şüpheli e-posta metnini aşağıdaki kutucuğa yapıştırın ve butona tıklayın.')

# Kullanıcıdan metin girmesi için bir alan oluşturuyoruz
user_input = st.text_area('E-posta Metni', height=200)

# Analiz butonu oluşturuyoruz
if st.button('Analiz Et'):
    if user_input:
        # Kullanıcının girdiği metni sayısala dönüştürüyoruz
        test_vector = vectorizer.transform([user_input])
        
        # Modelin tahminini alıyoruz
        prediction = model.predict(test_vector)
        prediction_proba = model.predict_proba(test_vector)

        # Sonucu ekrana yazdırıyoruz
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
