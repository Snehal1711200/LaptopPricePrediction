import pickle
import  numpy as np
import streamlit as st


# import model
pipe =pickle.load(open('pipe.pkl','rb'))
with open('df.pkl', 'rb') as file:
    df = pickle.load(file)

st.title('laptop price predictor')

#brand
company=st.selectbox('Brand',df['Company'].unique())

#type of laptop
type =st.selectbox('Type',df['TypeName'].unique())

#Ram
ram=st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

#weight
weight=st.number_input('Weight of the laptop')

#touchscreen
touchscreen=st.selectbox('Touchscreen',['No','Yes'])

#ips
ips=st.selectbox('IPS',['No','Yes'])

#screen size
screen_size=st.number_input('Screen Size')

#resolution
resolution=st.selectbox('Screen Resolution',['1920 x 1080', '1366 x 768' ,'1600 x 900','3840 x 2160','3200 x 1800',
                                             '2880 x 1800' ,'2560 x 1600','2560 x 1440','2304 x 1440'])

#cpu
cpu=st.selectbox('CPU',df['cpu_brand'].unique())

hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd=st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu=st.selectbox('GPU',df['Gpu brand'].unique())
os=st.selectbox('OS',df['os'].unique())

if st.button('predict price'):
    ppi=None
    if touchscreen =='Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips=='Yes':
        ips=1
    else:
        ips=0

    x_res= int(resolution.split('x')[0])
    y_res=int(resolution.split('x')[1])
    ppi=((x_res**2)+(y_res**2))**0.5/screen_size
    query=np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query=query.reshape(1,12)
    st.title('the predicted price of this configuration is ' + str(int(np.exp(pipe.predict(query)[0]))))

