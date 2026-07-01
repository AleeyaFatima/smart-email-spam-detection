import re
import streamlit as st
from typing import Dict, Any

# ==========================================
# DESIGN DECISION: Premium Custom UI Components
# Standard Streamlit widgets can look simple.
# To make this dashboard look like a real SaaS product (Vite/React aesthetic),
# we write python helper functions that inject custom HTML and CSS.
# These components render statistics cards with hover effects,
# prediction result badges, and spam keyword highlighting.
# ==========================================

def render_kpi_card(title: str, value: Any, subtitle: str, icon_class: str, card_type: str = "primary"):
    """
    Renders a premium, rounded card with modern styling, hover effects,
    and a custom FontAwesome icon.
    
    card_type options: 'primary', 'success', 'warning', 'info'
    """
    colors = {
        "primary": {"bg": "linear-gradient(135deg, #1e1e38 0%, #111122 100%)", "border": "#6366f1"},
        "success": {"bg": "linear-gradient(135deg, #0f2c22 0%, #081611 100%)", "border": "#10b981"},
        "warning": {"bg": "linear-gradient(135deg, #382415 0%, #1e1208 100%)", "border": "#f59e0b"},
        "info": {"bg": "linear-gradient(135deg, #122c3d 0%, #0a1721 100%)", "border": "#06b6d4"}
    }
    
    style = colors.get(card_type, colors["primary"])
    
    html = f"""
    <div class="kpi-card" style="
        background: {style['bg']};
        border-left: 5px solid {style['border']};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 10px 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: space-between;
    " onmouseover="this.style.transform='translateY(-3px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.3)';" 
      onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(0,0,0,0.2)';">
        <div>
            <p style="color: #9ca3af; margin: 0; font-size: 0.85rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">{title}</p>
            <h2 style="color: #ffffff; margin: 5px 0 0 0; font-size: 2rem; font-weight: 700;">{value}</h2>
            <p style="color: #6b7280; margin: 4px 0 0 0; font-size: 0.75rem;">{subtitle}</p>
        </div>
        <div style="
            background: rgba(255, 255, 255, 0.05);
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <i class="{icon_class}" style="color: {style['border']}; font-size: 1.5rem;"></i>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_prediction_badge(is_spam: bool, probability: float, confidence: float):
    """
    Renders a premium visual card showing prediction outcomes,
    completed with circular gauges and detailed micro-metrics.
    """
    if is_spam:
        badge_title = "SPAM DETECTED"
        color = "#ef4444"
        bg_gradient = "linear-gradient(135deg, #2d1616 0%, #1a0b0b 100%)"
        description = "This email matches spam triggers and phishing patterns. High risk."
        icon = "fa-solid fa-triangle-exclamation"
    else:
        badge_title = "CLEAN / SAFE"
        color = "#10b981"
        bg_gradient = "linear-gradient(135deg, #0e271f 0%, #071510 100%)"
        description = "This email is classified as legitimate (Ham). Low risk."
        icon = "fa-solid fa-circle-check"

    # Gauge background color
    gauge_bg = "#374151"
    
    html = f"""
    <div style="
        background: {bg_gradient};
        border: 1px solid {color}44;
        border-top: 4px solid {color};
        border-radius: 12px;
        padding: 24px;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin: 15px 0;
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 250px; padding-right: 15px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <i class="{icon}" style="color: {color}; font-size: 1.8rem;"></i>
                    <h3 style="margin: 0; font-size: 1.5rem; font-weight: 700; color: #ffffff; letter-spacing: 0.05em;">{badge_title}</h3>
                </div>
                <p style="color: #d1d5db; font-size: 0.95rem; line-height: 1.5; margin-bottom: 20px;">{description}</p>
                
                <div style="display: flex; gap: 20px; margin-top: 15px;">
                    <div>
                        <span style="color: #9ca3af; font-size: 0.8rem; text-transform: uppercase;">Confidence Score</span>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-top: 2px;">{confidence}%</div>
                    </div>
                    <div>
                        <span style="color: #9ca3af; font-size: 0.8rem; text-transform: uppercase;">Spam Probability</span>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-top: 2px;">{round(probability * 100, 1)}%</div>
                    </div>
                </div>
            </div>
            
            <div style="width: 140px; text-align: center; margin: 10px auto;">
                <div style="
                    position: relative;
                    width: 110px;
                    height: 110px;
                    border-radius: 50%;
                    background: radial-gradient(circle, #0f172a 60%, transparent 61%), conic-gradient({color} {probability*360}deg, {gauge_bg} 0deg);
                    margin: 0 auto;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
                ">
                    <span style="font-size: 1.4rem; font-weight: 700; color: #ffffff;">{round(probability * 100)}%</span>
                </div>
                <div style="margin-top: 8px; font-size: 0.75rem; color: #9ca3af; font-weight: 500;">SPAM PROBABILITY</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def highlight_spam_keywords(text: str) -> str:
    """
    Identifies common high-priority spam words and wraps them in HTML highlight badges.
    This creates an extremely premium visual analyzer tool.
    """
    # Key spam indicator terms
    spam_triggers = [
        r"\bwinner\b", r"\bwon\b", r"\bfree\b", r"\bprize\b", r"\bcash\b", r"\bclaim\b",
        r"\burgent\b", r"\bselected\b", r"\brefinance\b", r"\bviagra\b", r"\bcialis\b",
        r"\bgift card\b", r"\bclick here\b", r"\bclick now\b", r"\bmake money\b", r"\bdouble income\b",
        r"\bmillions?\b", r"\bbillion\b", r"\bcredit card\b", r"\baccount locked\b", r"\bsuspicious activity\b"
    ]
    
    highlighted_text = text
    for pattern in spam_triggers:
        # Regex replacement ignoring case, wrapping matching word in a beautiful glowing red span
        highlighted_text = re.sub(
            pattern,
            lambda match: f'<span style="background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 4px; padding: 2px 6px; color: #f87171; font-weight: 600;">{match.group(0)}</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )
        
    return highlighted_text


def render_keyword_analyzer(raw_text: str):
    """
    Renders an interactive keyword inspector showing highlighted words inside a custom box.
    """
    highlighted = highlight_spam_keywords(raw_text)
    
    html = f"""
    <div style="
        background: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        min-height: 100px;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
    ">
        <p style="margin: 0; white-space: pre-wrap;">{highlighted}</p>
    </div>
    <div style="margin-top: 8px; font-size: 0.75rem; color: #9ca3af; display: flex; align-items: center; gap: 5px;">
        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #ef4444;"></span>
        Red highlighted text represents words commonly found in spam datasets.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
