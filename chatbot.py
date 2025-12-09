from dotenv import load_dotenv
import streamlit as st

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

# ==========================
# 1. Ładowanie zmiennych środowiskowych
# ==========================
load_dotenv()

# ==========================
# 2. Konfiguracja aplikacji Streamlit
# ==========================
st.set_page_config(
    page_title="Multi-Model Chatbot",
    page_icon="🤖",
    layout="centered",
)
st.title("🤖 Multi-Model Chatbot")

st.write(
    "Wybierz *provider* i *model*, a potem zacznij rozmowę. "
    "Historia czatu jest zachowywana w `session_state`."
)

# ==========================
# 3. Konfiguracja providerów i modeli
# ==========================

PROVIDERS = {
    "OpenAI": [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-3.5-turbo",
    ],
    "Gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    "Groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ],
    "Ollama": [
        "llama3.1",
        "gemma3",
    ],
}


def get_llm(provider: str, model_name: str):
    """
    Zwraca obiekt ChatModel z LangChain
    w zależności od wybranego providera.
    """
    if provider == "OpenAI":
        return ChatOpenAI(model=model_name, temperature=0.2)
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    elif provider == "Groq":
        return ChatGroq(model=model_name, temperature=0.2)
    elif provider == "Ollama":
        # wymagany działający lokalnie serwer Ollama
        return ChatOllama(model=model_name, temperature=0.2)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ==========================
# 4. Sidebar – wybór providera i modelu
# ==========================

with st.sidebar:
    st.header("⚙️ Ustawienia modelu")

    provider = st.selectbox(
        "Wybierz providera:",
        options=list(PROVIDERS.keys()),
        index=2,  # domyślnie Groq
    )

    model_name = st.selectbox(
        "Wybierz model:",
        options=PROVIDERS[provider],
    )

    st.markdown(
        f"**Aktualny provider:** `{provider}`  \n"
        f"**Aktualny model:** `{model_name}`"
    )

# ==========================
# 5. Inicjalizacja historii czatu
# ==========================

if "chat_history" not in st.session_state:
    # przechowujemy listę LangChain messages:
    st.session_state.chat_history = []

if "current_provider" not in st.session_state:
    st.session_state.current_provider = provider

if "current_model" not in st.session_state:
    st.session_state.current_model = model_name

# Reset historii, jeśli zmienimy providera lub model
if (
    provider != st.session_state.current_provider
    or model_name != st.session_state.current_model
):
    st.session_state.chat_history = []
    st.session_state.current_provider = provider
    st.session_state.current_model = model_name
    st.info("Zmieniono providera lub model – historia czatu została wyczyszczona.")

# ==========================
# 6. Wyświetlanie historii czatu
# ==========================

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        # pomijamy SystemMessage przy wyświetlaniu
        continue

    with st.chat_message(role):
        st.markdown(message.content)

# ==========================
# 7. Inicjalizacja LLM dla aktualnego providera
# ==========================

llm = get_llm(provider, model_name)

# Stały system prompt – możesz dopasować pod zadanie
SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Answer in the language of the user."
)

# ==========================
# 8. Pole inputu użytkownika
# ==========================

user_prompt = st.chat_input("Napisz wiadomość do wybranego modelu...")

if user_prompt:
    # 1) pokaż wiadomość użytkownika
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2) dopisz do historii
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))

    # 3) przygotuj listę wiadomości do modelu:
    messages_for_model = [
        SystemMessage(content=SYSTEM_PROMPT),
        *st.session_state.chat_history,
    ]

    # 4) wyślij do LLM
    response = llm.invoke(messages_for_model)

    # 5) zapisz odpowiedź w historii
    st.session_state.chat_history.append(AIMessage(content=response.content))

    # 6) wyświetl odpowiedź
    with st.chat_message("assistant"):
        st.markdown(response.content)


    # Workflow:
    # user_prompt -> zapis do historii -> wysłanie historii do llm ->
    # odpowiedź -> zapis odpowiedzi do historii -> wyświetlenie odpowiedzi
