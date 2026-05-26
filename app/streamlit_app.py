import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Adventure • Premium", page_icon="🌍", layout="wide")

# ===================== PREMIUM DARK THEME =====================
st.markdown("""
<style>
    .stApp { background-color: #0a0c14; color: #f0f0f0; }
    .main-header { 
        font-size: 3.8rem; 
        font-weight: 800; 
        color: #f4d35e; 
        text-align: center; 
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ===================== LOAD MODELS =====================
@st.cache_resource
def load_models():
    BASE_DIR = Path(__file__).parent.parent
    model = joblib.load(BASE_DIR / "models" / "final_clean_model.joblib")
    encoder = joblib.load(BASE_DIR / "models" / "final_clean_encoder.joblib")
    le = joblib.load(BASE_DIR / "models" / "final_clean_label_encoder.joblib")
    return model, encoder, le

model, encoder, label_encoder = load_models()

# ===================== PLACE DESCRIPTIONS =====================
place_descriptions = {
    "Nairobi": "Nairobi, Kenya’s vibrant capital, blends urban energy with nature. It’s one of the few cities in the world with a national park right next to it.",
    "Maasai Mara": "The Maasai Mara is world-famous for its incredible wildlife and the Great Migration. A must-visit for any safari lover.",
    "Cape Town": "Cape Town is South Africa’s most scenic city, surrounded by mountains and ocean. Home to Table Mountain and beautiful beaches.",
    "Victoria Falls": "One of the largest and most spectacular waterfalls in the world, located on the border of Zambia and Zimbabwe.",
    "Kruger National Park": "South Africa’s largest game reserve and one of the best places in Africa to see the Big Five.",
    "Bali": "Bali is Indonesia’s most popular island, known for its beaches, temples, rice terraces, and spiritual atmosphere.",
    "Dubai": "A futuristic city of luxury, innovation, and desert adventures. Famous for the Burj Khalifa and world-class experiences.",
    "Tokyo": "Tokyo perfectly balances ultra-modern technology with deep-rooted traditions. A city full of contrasts and excitement.",
    "Bangkok": "Thailand’s capital is a vibrant mix of street food, temples, markets, and modern skyscrapers.",
    "Singapore": "A clean, green, and futuristic city-state known for its food, gardens, and efficient lifestyle.",
    "Maldives": "A tropical paradise made of crystal-clear waters and overwater villas. Perfect for honeymoons and relaxation.",
    "Paris": "The City of Light is famous for its art, fashion, cuisine, and iconic landmarks like the Eiffel Tower.",
    "Santorini": "A stunning Greek island known for its white-washed buildings, blue domes, and magical sunsets.",
    "Rome": "Rome is a living museum filled with ancient history, incredible food, and beautiful architecture.",
    "Barcelona": "A vibrant Spanish city known for Gaudí’s architecture, beaches, and lively street culture.",
    "Queenstown": "New Zealand’s adventure capital, surrounded by mountains and lakes. Perfect for thrill-seekers.",
    "Sydney": "Australia’s largest city, famous for the Sydney Opera House and beautiful beaches.",
    "Istanbul": "A magical city where East meets West. Famous for its historic mosques, bazaars, and rich culture.",
}

def get_place_intro(city, country):
    return place_descriptions.get(city, f"{city} is a beautiful destination in {country} with unique culture and experiences.")

# ===================== SIDEBAR NAVIGATION =====================
st.sidebar.title("🌍 ADVENTURE")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔮 Plan Your Trip", "📜 Trip History"])

# ===================== PAGE 1: HOME =====================
if page == "🏠 Home":
    st.markdown("<h1 class='main-header'>ADVENTURE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.4rem;'>Premium AI-Powered Travel Planning</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## Welcome to Adventure")
    st.write("Use AI to plan better trips with personalized predictions, best time suggestions, cost estimates, and destination insights.")

    st.markdown("### What You Can Do")
    col1, col2, col3 = st.columns(3)
    with col1: st.info("**Smart Predictions**")
    with col2: st.info("**Best Time Suggestions**")
    with col3: st.info("**Cost & Packages**")

# ===================== PAGE 2: PLAN YOUR TRIP =====================
elif page == "🔮 Plan Your Trip":
    st.markdown("<h1 class='main-header'>Plan Your Trip</h1>", unsafe_allow_html=True)
    st.markdown("---")

    with st.form("trip_form"):
        col1, col2 = st.columns(2)

        with col1:
            travel_dates = st.date_input("Travel Dates", value=(date(2026, 7, 1), date(2026, 7, 10)))
            no_of_days = st.number_input("How many days will you stay?", min_value=2, max_value=30, value=7, step=1)
            total_members = st.number_input("Total Number of Members", min_value=1, max_value=20, value=2)
            adults = st.number_input("Number of Adults", min_value=1, max_value=20, value=2)
            kids = st.number_input("Number of Kids", min_value=0, max_value=10, value=0)

            continent = st.selectbox("Continent", ["Africa", "Asia", "Europe", "America", "Australia & Oceania"])
            country = st.text_input("Country", "Kenya")

        with col2:
            city = st.text_input("City / Destination", "Nairobi")
            attraction = st.text_input("Main Attraction", "Maasai Mara")
            attr_type = st.selectbox("Experience Type", ["Wildlife & Safari", "Beach", "Adventure", "Cultural", "Business"])

        submitted = st.form_submit_button("Create My Trip Plan")

    if submitted:
        month = travel_dates[0].month if isinstance(travel_dates, tuple) else travel_dates.month

        input_data = pd.DataFrame({
            'VisitMonth': [month],
            'Continent': [continent],
            'Country': [country],
            'CityName': [city],
            'Attraction': [attraction],
            'AttractionType': [attr_type]
        })

        encoded = encoder.transform(input_data)
        pred = model.predict(encoded)[0]

        # ✅ Fixed: Manual mapping so it shows "Business" or "Leisure"
        if pred == 1:
            travel_type = "Business"
        else:
            travel_type = "Leisure"

        st.markdown("---")
        st.markdown("## Your Personalized Trip Plan")

        # Travel Style + Group Info
        st.markdown(f"<div style='background-color:#1a1d2e; padding:20px; border-radius:12px;'><h3>Your Travel Style: <span style='color:#f4d35e;'>{travel_type}</span></h3><p><strong>Group:</strong> {total_members} members ({adults} adults + {kids} kids)</p></div>", unsafe_allow_html=True)

        # About Destination
        st.markdown("### 📍 About Your Destination")
        st.write(get_place_intro(city, country))

        # Dynamic Itinerary
        st.markdown(f"### 🗓️ Suggested {no_of_days}-Day Itinerary")
        if travel_type == "Leisure":
            st.write(f"""
            **Day 1:** Arrival + Check-in + Light exploration  
            **Day 2 to {no_of_days-1}:** Main activities (Safari, Beach, or Adventure)  
            **Day {no_of_days}:** Relaxed day + Departure
            """)
        else:
            st.write(f"""
            **Day 1:** Arrival + Check-in + Business meetings  
            **Day 2 to {no_of_days-1}:** Work + Evening leisure  
            **Day {no_of_days}:** Final meetings + Departure
            """)

        # Packages
        st.markdown("### 📦 Recommended Packages")
        if travel_type == "Leisure":
            st.success("**🌿 Leisure Explorer Package** — Includes stay + activities")
            st.info("**🏕️ Family Adventure Package** — Good for groups with kids")
        else:
            st.success("**💼 Business Comfort Package** — Includes hotel + transfers")

        # Best Stays
        st.markdown("### 🏨 Recommended Stays")
        if travel_type == "Leisure":
            st.write("**Luxury:** Four Seasons / Angama Mara\n**Comfort:** Mid-range safari lodges")
        else:
            st.write("**Business Friendly:** Modern hotels with good WiFi (Hilton, Marriott, etc.)")

        # Cost
        stay_cost = no_of_days * 120 * total_members
        food_cost = no_of_days * 60 * total_members

        st.markdown("### 💰 Estimated Total Cost")
        st.write(f"**Stay:** ${stay_cost} – ${stay_cost + 500}  |  **Travel:** $300 – $750  |  **Food & Local:** ${food_cost}")

# ===================== PAGE 3: TRIP HISTORY =====================
elif page == "📜 Trip History":
    st.markdown("<h1 class='main-header'>Trip History</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("This feature will show your previous trip analyses. (Coming soon)")

    st.markdown("### Recent Trips (Demo)")
    st.write("• June 2025 → Maasai Mara (Leisure) — 7 days")
    st.write("• March 2025 → Dubai (Business) — 4 days")
    st.write("• December 2024 → Bali (Leisure) — 10 days")

st.markdown("---")
st.caption("Adventure • Premium Travel Planning Experience")