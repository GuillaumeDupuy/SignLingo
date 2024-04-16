import streamlit as st
from gtts import gTTS, lang
from googletrans import Translator, LANGUAGES
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image

# ---------------------------------------------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="SignLingo", page_icon="👋", initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------------------------------------------
st.title("SignLingo")

st.write('<br>', unsafe_allow_html=True)
st.write("### SignLingo is a web app that translates sign language to text & speech.")
st.write('#### Bridging Silence with Words and a World where Every Sign Counts')
st.write('<br>', unsafe_allow_html=True)

list_langues = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'ug': 'uyghur',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu',
}

# ---------------------------------------------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------------------------------------------

def detect_lang(texte):
    """
    Detect the language of the text
    """
    detect = Translator()
    detect_lang = detect.detect(texte)
    return detect_lang.lang

def translate_text(texte, langue_cible):
    """
    Translate the text to the target language
    """
    translator = Translator()
    traduction = translator.translate(texte, dest=langue_cible)
    return traduction.text

def text_to_audio(texte, langue):
    """
    Convert the text to speech
    """
    tts = gTTS(text=texte, lang=langue, slow=False)
    tts.save("output.mp3")

# ---------------------------------------------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------------------------------------------

st.write("### Record your sign language video:")

# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )

# class VideoProcessor:
#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
        
#         # vision processing
#         flipped = img[:, ::-1, :]

#         # model processing
#         im_pil = Image.fromarray(flipped)
#         results = st.model(im_pil, size=112)
#         bbox_img = np.array(results.render()[0])

#         return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")

# webrtc_ctx = webrtc_streamer(
#     key="WYH",
#     mode=WebRtcMode.SENDRECV,
#     rtc_configuration=RTC_CONFIGURATION,
#     video_processor_factory=VideoProcessor,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=False,
# )

webrtc_streamer(key="example")

# col1, col2 = st.columns([1,1])

# with col1:
#     record_button_pressed = st.button("Record", key="record_button")

# with col2:
#     stop_button_pressed = st.button("Stop", key="stop_button")

# if record_button_pressed:

#     cap = cv2.VideoCapture(0)
#     frame_placeholder = st.empty()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.write("Video Capture Ended")
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_placeholder.image(frame,channels="RGB")
#         if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
#             break
#     cap.release()
#     cv2.destroyAllWindows()

#     img_file_buffer = st.camera_input("Take a picture")

#     if img_file_buffer is not None:
#         bytes_data = img_file_buffer.getvalue()
#         cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#         st.write(type(cv2_img))

#         st.write(cv2_img.shape)

st.write('<br>', unsafe_allow_html=True)

st.write("### Translate your text:")

texte = st.text_input("Type your text here :", "How are you ?")
texte = texte.lower()

lang_detect = detect_lang(texte)
# Recover the language name from the language code
langue_detect = list_langues[lang_detect]

st.write("### Your text is in ")
st.write(langue_detect)

# Select the language to translate to (default: french)
st.write('<br>', unsafe_allow_html=True)
langue = st.selectbox("Select a language to translate to :", list(list_langues.values()), index=list(list_langues.values()).index('french'))
# Recover the language code from the language name
langue_code = list(list_langues.keys())[list(list_langues.values()).index(langue)]

texte_traduit = translate_text(texte, langue_code)

st.write("### Your text translated to " + langue + " is : ")
st.write(texte_traduit)
st.write('<br>', unsafe_allow_html=True)


# Check if the selected language is supported by gTTS
if langue_code in lang.tts_langs():
    st.write("### Your text translated to speech:")

    text_to_audio(texte_traduit, langue_code)
    audio_file = open('output.mp3', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
else:
    st.write("### Sorry, the selected language is not supported for text-to-speech.")

# st.write('<br>', unsafe_allow_html=True)
# st.write(lang.tts_langs())
# st.write(LANGUAGES)