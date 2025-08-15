import streamlit as st
import google.generativeai as genai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import os
import logging
import requests
import folium
from streamlit_folium import folium_static
from geopy.distance import geodesic
import datetime
import json

# Page configuration
st.set_page_config(
    page_title="MediCare AI Assistant", 
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .patient-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .symptom-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .medicine-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .process-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up Gemini API
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAw1MHEgBpMZrWYfGcjw0v4dIqHNXYYKNo'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def query_healthcare_assistant(patient_info):
    # Create a more specific and detailed prompt
    symptoms_text = patient_info.get('symptoms', 'No symptoms provided')
    
    prompt = f"""
    You are a medical AI assistant. Analyze this patient case and provide detailed recommendations:
    
    PATIENT PROFILE:
    Name: {patient_info['name']}
    Age: {patient_info['age']} years
    Gender: {patient_info['gender']}
    Symptoms: {symptoms_text}
    Severity Level: {patient_info['severity']}/10
    Duration: {patient_info['duration']}
    Medical History: {patient_info.get('medical_history', 'None provided')}
    Current Medications: {patient_info.get('current_meds', 'None')}
    Known Allergies: {patient_info.get('allergies', 'None')}
    
    Please provide a comprehensive medical analysis with specific details:
    
    POSSIBLE CONDITIONS:
    Based on the symptoms described, list 3-4 most likely medical conditions with brief explanations.
    
    RECOMMENDED MEDICINES:
    Suggest appropriate over-the-counter or commonly prescribed medications:
    - Medicine name, typical dosage, frequency, and purpose
    - Consider patient's age and any contraindications
    
    TREATMENT PROCESS:
    Provide step-by-step treatment approach:
    1. Immediate relief measures
    2. Short-term treatment plan (1-2 weeks)
    3. If needed, long-term management
    
    LIFESTYLE RECOMMENDATIONS:
    Suggest specific lifestyle changes:
    - Dietary modifications
    - Activity and rest recommendations
    - Environmental factors to consider
    
    WHEN TO SEEK IMMEDIATE CARE:
    List specific warning signs that require emergency attention
    
    FOLLOW-UP SCHEDULE:
    Recommend when to see a healthcare provider
    
    IMPORTANT DISCLAIMERS:
    Include standard medical disclaimers about AI advice limitations
    
    Please be specific and practical in your recommendations while maintaining medical safety.
    """
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        # Add generation config for better responses
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        response = model.generate_content(prompt, generation_config=generation_config)
        
        if response and hasattr(response, 'text') and response.text:
            return response.text
        elif response and response.parts:
            # Try to extract text from parts
            text_parts = []
            for part in response.parts:
                if hasattr(part, 'text'):
                    text_parts.append(part.text)
            if text_parts:
                return '\n'.join(text_parts)
        
        # If we still don't have content, provide a fallback
        logging.error("No valid response text found.")
        return generate_fallback_response(patient_info)
        
    except Exception as e:
        logging.error(f"Error in Gemini API call: {e}")
        return generate_fallback_response(patient_info)

def generate_fallback_response(patient_info):
    """Generate a basic medical response when AI is unavailable"""
    return f"""
    POSSIBLE CONDITIONS:
    Based on the symptoms described for {patient_info['name']}, here are some general considerations:
    - Common viral or bacterial infections
    - Stress-related conditions
    - Nutritional deficiencies
    - Environmental factors
    
    RECOMMENDED MEDICINES:
    - Rest and adequate hydration
    - Over-the-counter pain relievers if needed (as per package instructions)
    - Multivitamin supplements
    - Consult pharmacist for symptom-specific medications
    
    TREATMENT PROCESS:
    1. Get adequate rest and sleep (7-9 hours daily)
    2. Stay well-hydrated with water and clear fluids
    3. Monitor symptoms for any changes
    4. Consider gentle exercise if feeling up to it
    
    LIFESTYLE RECOMMENDATIONS:
    - Maintain a balanced diet with fruits and vegetables
    - Reduce stress through relaxation techniques
    - Avoid smoking and limit alcohol consumption
    - Maintain good hygiene practices
    
    WHEN TO SEEK IMMEDIATE CARE:
    - High fever (over 103°F/39.4°C)
    - Difficulty breathing or chest pain
    - Severe dehydration
    - Symptoms that worsen rapidly
    - Any concerning changes in condition
    
    FOLLOW-UP SCHEDULE:
    - If symptoms persist beyond 1 week, consult a healthcare provider
    - Schedule routine check-up within 2-4 weeks
    - Immediate care if any emergency signs develop
    
    IMPORTANT DISCLAIMERS:
    - This is general health information only
    - Not a substitute for professional medical advice
    - Always consult qualified healthcare providers
    - Seek emergency care when in doubt
    """

def create_enhanced_pdf(patient_info, medical_report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=30,
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#A23B72'),
        spaceBefore=20,
        spaceAfter=12
    )
    
    # Title
    story.append(Paragraph("MEDICARE AI - MEDICAL CONSULTATION REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information Table
    patient_data = [
        ['Patient Name:', patient_info['name']],
        ['Age:', f"{patient_info['age']} years"],
        ['Gender:', patient_info['gender']],
        ['Consultation Date:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ['Symptom Severity:', f"{patient_info['severity']}/10"],
        ['Duration:', patient_info['duration']],
        ['Location:', patient_info['location']]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F8FF')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    story.append(Paragraph("Patient Information", subtitle_style))
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Medical Report
    story.append(Paragraph("Medical Analysis & Recommendations", subtitle_style))
    
    # Split report into sections and format
    sections = medical_report.split('\n\n')
    for section in sections:
        if section.strip():
            story.append(Paragraph(section.replace('\n', '<br/>'), styles['Normal']))
            story.append(Spacer(1, 12))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph("Generated by MediCare AI Assistant | This is not a substitute for professional medical advice", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def get_coordinates(address):
    url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
    headers = {"User-Agent": "HealthcareAssistant/1.0"}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except:
        pass
    return None, None

def find_nearby_places(lat, lon, place_type, radius=5000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="{place_type}"](around:{radius},{lat},{lon});
      way["amenity"="{place_type}"](around:{radius},{lat},{lon});
      relation["amenity"="{place_type}"](around:{radius},{lat},{lon});
    );
    out center;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        data = response.json()
        return data['elements']
    except:
        return []

# Main App
st.markdown('<div class="main-header"><h1>🏥 MediCare AI Assistant</h1><p>Your Personal Healthcare Companion</p></div>', unsafe_allow_html=True)

# Initialize session state
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Sidebar for patient information
with st.sidebar:
    st.header("📋 Patient Information")
    
    with st.form("patient_form"):
        name = st.text_input("👤 Full Name*", value=st.session_state.patient_data.get('name', ''))
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("🎂 Age*", min_value=1, max_value=120, value=st.session_state.patient_data.get('age', 25))
        with col2:
            gender = st.selectbox("⚥ Gender*", ["Male", "Female", "Other"], index=0)
        
        st.subheader("🏥 Medical Details")
        symptoms = st.text_area("🩺 Current Symptoms*", 
                               value=st.session_state.patient_data.get('symptoms', ''),
                               help="Describe your symptoms in detail",
                               height=100)
        
        severity = st.slider("📊 Symptom Severity (1-10)", 1, 10, 5)
        duration = st.selectbox("⏱️ Duration", 
                               ["Less than 1 day", "1-3 days", "4-7 days", "1-2 weeks", "2-4 weeks", "More than 1 month"])
        
        medical_history = st.text_area("📚 Medical History", 
                                     value=st.session_state.patient_data.get('medical_history', 'None'),
                                     height=80)
        current_meds = st.text_area("💊 Current Medications", 
                                  value=st.session_state.patient_data.get('current_meds', 'None'),
                                  height=80)
        allergies = st.text_input("🚫 Allergies", value=st.session_state.patient_data.get('allergies', 'None'))
        
        st.subheader("📍 Location")
        location = st.text_input("🗺️ Your Location*", 
                                value=st.session_state.patient_data.get('location', ''),
                                help="Enter city, state, or full address")
        search_radius = st.slider("🔍 Search Radius (km)", 1, 20, 5)
        
        submitted = st.form_submit_button("🔍 Analyze & Find Care", type="primary")
        
        if submitted:
            if name and symptoms and location:
                st.session_state.patient_data = {
                    'name': name,
                    'age': age,
                    'gender': gender,
                    'symptoms': symptoms,
                    'severity': severity,
                    'duration': duration,
                    'medical_history': medical_history,
                    'current_meds': current_meds,
                    'allergies': allergies,
                    'location': location,
                    'search_radius': search_radius * 1000
                }
                st.session_state.analysis_complete = False
            else:
                st.error("Please fill in all required fields marked with *")

# Main content area
if st.session_state.patient_data:
    patient_info = st.session_state.patient_data
    
    # Patient summary card
    st.markdown(f"""
    <div class="patient-card">
        <h3>👤 Patient: {patient_info['name']}</h3>
        <p><strong>Age:</strong> {patient_info['age']} years | <strong>Gender:</strong> {patient_info['gender']}</p>
        <p><strong>Location:</strong> {patient_info['location']}</p>
        <p><strong>Severity:</strong> {patient_info['severity']}/10 | <strong>Duration:</strong> {patient_info['duration']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.analysis_complete:
        with st.spinner("🔬 Analyzing symptoms and finding nearby healthcare facilities..."):
            try:
                # Get medical analysis
                medical_report = query_healthcare_assistant(patient_info)
                st.session_state.medical_report = medical_report
                
                # Debug: Show if we got a response
                if medical_report and len(medical_report.strip()) > 50:
                    st.success("✅ Medical analysis completed successfully!")
                else:
                    st.warning("⚠️ Medical analysis generated but may be incomplete. Using fallback recommendations.")
                
                # Get location data
                lat, lon = get_coordinates(patient_info['location'])
                if lat and lon:
                    hospitals = find_nearby_places(lat, lon, "hospital", patient_info['search_radius'])
                    pharmacies = find_nearby_places(lat, lon, "pharmacy", patient_info['search_radius'])
                    st.session_state.location_data = {
                        'lat': lat,
                        'lon': lon,
                        'hospitals': hospitals,
                        'pharmacies': pharmacies
                    }
                    st.success(f"📍 Found location: {patient_info['location']} ({len(hospitals)} hospitals, {len(pharmacies)} pharmacies nearby)")
                else:
                    st.session_state.location_data = None
                    st.warning("⚠️ Could not find location coordinates. Map features will be unavailable.")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.session_state.medical_report = generate_fallback_response(patient_info)
                st.session_state.location_data = None
            
            st.session_state.analysis_complete = True
    
    if st.session_state.analysis_complete:
        # Display medical analysis
        st.markdown('<div class="symptom-card"><h3>🩺 Medical Analysis & Recommendations</h3></div>', unsafe_allow_html=True)
        
        # Parse and display the medical report in structured format
        report_text = st.session_state.medical_report
        
        # First, let's display the raw report for debugging if sections aren't found
        if not any(keyword in report_text for keyword in ['POSSIBLE CONDITIONS:', 'RECOMMENDED MEDICINES:', 'TREATMENT PROCESS:']):
            st.markdown("### 🩺 Complete Medical Analysis")
            st.markdown(f"<div class='medicine-card' style='white-space: pre-wrap;'>{report_text}</div>", unsafe_allow_html=True)
        else:
            # Parse structured sections
            sections = {
                'POSSIBLE CONDITIONS:': ('🔍 Possible Conditions', 'medicine-card'),
                'RECOMMENDED MEDICINES:': ('💊 Recommended Medicines', 'medicine-card'),
                'TREATMENT PROCESS:': ('🏥 Treatment Process', 'process-card'),
                'LIFESTYLE RECOMMENDATIONS:': ('🌱 Lifestyle Recommendations', 'success-box'),
                'WHEN TO SEEK IMMEDIATE CARE:': ('🚨 When to Seek Immediate Care', 'warning-box'),
                'FOLLOW-UP SCHEDULE:': ('📅 Follow-up Schedule', 'info'),
                'IMPORTANT DISCLAIMERS:': ('⚖️ Important Disclaimers', 'warning')
            }
            
            for section_key, (title, style_class) in sections.items():
                if section_key in report_text:
                    st.markdown(f"### {title}")
                    
                    # Extract content for this section
                    start_idx = report_text.find(section_key) + len(section_key)
                    
                    # Find the next section or end of text
                    next_section_idx = len(report_text)
                    for other_key in sections.keys():
                        if other_key != section_key:
                            other_idx = report_text.find(other_key, start_idx)
                            if other_idx != -1 and other_idx < next_section_idx:
                                next_section_idx = other_idx
                    
                    content = report_text[start_idx:next_section_idx].strip()
                    
                    # Format content based on section
                    if style_class == 'warning-box' and section_key == 'WHEN TO SEEK IMMEDIATE CARE:':
                        st.markdown(f"<div class='{style_class}'>⚠️ <strong>Emergency Signs:</strong><br>{content.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                    elif style_class == 'info':
                        st.info(content)
                    elif style_class == 'warning':
                        st.warning(content)
                    else:
                        st.markdown(f"<div class='{style_class}' style='white-space: pre-wrap;'>{content}</div>", unsafe_allow_html=True)
        
        # PDF Download
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            pdf_buffer = create_enhanced_pdf(patient_info, st.session_state.medical_report)
            st.download_button(
                label="📄 Download Detailed Report (PDF)",
                data=pdf_buffer,
                file_name=f"medical_report_{patient_info['name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                type="primary"
            )
        
        with col2:
            if st.button("🔄 New Consultation", type="secondary"):
                st.session_state.patient_data = {}
                st.session_state.analysis_complete = False
                st.rerun()
        
        # Location-based services
        if st.session_state.location_data:
            location_data = st.session_state.location_data
            st.markdown("---")
            st.markdown("### 🗺️ Nearby Healthcare Facilities")
            
            # Create map
            m = folium.Map(location=[location_data['lat'], location_data['lon']], zoom_start=13)
            folium.Marker(
                [location_data['lat'], location_data['lon']], 
                popup="📍 Your Location", 
                icon=folium.Icon(color='red', icon='home')
            ).add_to(m)
            
            # Add hospitals
            for hospital in location_data['hospitals']:
                if 'lat' in hospital:
                    name = hospital.get('tags', {}).get('name', 'Hospital')
                    folium.Marker(
                        [hospital['lat'], hospital['lon']], 
                        popup=f"🏥 {name}",
                        icon=folium.Icon(color='blue', icon='plus-sign')
                    ).add_to(m)

            # Add pharmacies
            for pharmacy in location_data['pharmacies']:
                if 'lat' in pharmacy:
                    name = pharmacy.get('tags', {}).get('name', 'Pharmacy')
                    folium.Marker(
                        [pharmacy['lat'], pharmacy['lon']], 
                        popup=f"💊 {name}",
                        icon=folium.Icon(color='green', icon='shopping-cart')
                    ).add_to(m)

            folium_static(m)
            
            # Display facilities list
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏥 Nearby Hospitals")
                hospital_count = 0
                for hospital in location_data['hospitals'][:5]:
                    if 'tags' in hospital and 'name' in hospital['tags'] and hospital_count < 5:
                        distance = geodesic((location_data['lat'], location_data['lon']), 
                                          (hospital['lat'], hospital['lon'])).km
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{hospital['tags']['name']}</strong><br>
                            📍 {distance:.1f} km away
                        </div>
                        """, unsafe_allow_html=True)
                        hospital_count += 1
            
            with col2:
                st.markdown("#### 💊 Nearby Pharmacies")
                pharmacy_count = 0
                for pharmacy in location_data['pharmacies'][:5]:
                    if 'tags' in pharmacy and 'name' in pharmacy['tags'] and pharmacy_count < 5:
                        distance = geodesic((location_data['lat'], location_data['lon']), 
                                          (pharmacy['lat'], pharmacy['lon'])).km
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{pharmacy['tags']['name']}</strong><br>
                            📍 {distance:.1f} km away
                        </div>
                        """, unsafe_allow_html=True)
                        pharmacy_count += 1

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to MediCare AI Assistant
    
    Get personalized medical recommendations based on your symptoms, age, and medical history.
    
    ### ✨ Features:
    - 🩺 **Comprehensive Symptom Analysis** - AI-powered medical assessment
    - 💊 **Personalized Medicine Recommendations** - Based on your profile
    - 🏥 **Treatment Process Guidance** - Step-by-step care instructions
    - 🗺️ **Nearby Healthcare Facilities** - Find hospitals and pharmacies
    - 📄 **Detailed PDF Reports** - Download your consultation summary
    - 🚨 **Emergency Guidelines** - Know when to seek immediate care
    
    ### 🚀 How to Get Started:
    1. Fill out the patient information form in the sidebar ➡️
    2. Describe your symptoms in detail
    3. Enter your location for nearby facility search
    4. Click "Analyze & Find Care" to get your personalized recommendations
    
    ---
    
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> This AI assistant provides general information and recommendations. 
        Always consult with qualified healthcare professionals for proper diagnosis and treatment. 
        In case of emergency, call your local emergency services immediately.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏥 MediCare AI Assistant | Powered by Google Gemini AI</p>
    <p>📍 Location services by OpenStreetMap | 🗺️ Interactive maps by Folium</p>
</div>
""", unsafe_allow_html=True)