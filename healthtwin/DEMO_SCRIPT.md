# HealthTwin Pitch & Demo Script

This script provides a structured, 3-minute live demonstration flow for the hackathon judges.

## Pitch (1 Minute)

**The Problem:**
> "Modern wearables give us hundreds of raw datapoints every day, but zero contextual intelligence. When a wearable says 'Your heart rate is 85', it doesn't know if you're recovering from a workout, stressed at work, or getting sick."

**Our Solution:**
> "We built **HealthTwin**, an AI-powered digital twin. Instead of comparing you to population averages, HealthTwin builds a 30-day structural model of *you*. It processes multi-modal data (HRV, Sleep, EDA) through a custom rolling feature engine, calculates your personal baselines dynamically, and uses XGBoost models to predict exact cardiovascular, stress, and sleep risks in real-time."

**The Missing Piece (Showcase):**
> "But predicting risk isn't enough. People need to know *why*, and *what to do*. Let me show you the live dashboard."

---

## Live Demo (2 Minutes)

*(Start screen share with `streamlit run app/streamlit_app.py` already open)*

### 1. The Real-time Engine
> "Here you can see the engine running. On the left, we are spooling through months of continuous wearable data, calculating 170+ trailing metrics on-the-fly and pushing it into our three XGBoost risk models."
*Action: Click **▶️ Play** in the sidebar. Let it run for 10 seconds, then pause when Cardiac Risk pulses yellow or red.*

### 2. SHAP Explainability (XAI)
> "Notice how Cardiac Risk just spiked to High. Most apps just leave you guessing. HealthTwin uses Shapley mathematics underneath the hood."
*Action: Point to the SHAP Driver Analysis waterfall chart on the right.*
> "This chart proves that the model isn't a black box. It explicitly shows that a massive drop in HRV combined with a poor night's sleep actively pushed the prediction boundary up by 15 points."

### 3. AI Health Brief (The LLM)
> "To translate this math for the user, we feed the SHAP impacts and the Z-score deviations into an LLM context."
*Action: Point to the 'AI Health Brief' component.*
> "The engine instantly outputs a natural language summary and an immediate physical action step."

### 4. DiCE Sandbox & Counterfactuals
> "Finally, let's look at the Sandbox. What if the user wants to lower this risk immediately?"
*Action: On the sidebar, click **Get AI Suggestion**. It will recommend adding sleep or activity.*
> "HealthTwin's counterfactual engine runs millions of hypothetical perturbations. If I simulate adding 2 hours of sleep right now..."
*Action: Change the Sleep Hours in the What-If Sandbox and click **Apply Scenario**.*
> "...you can see the risk delta instantly drops by 20%. The digital twin proved exactly what behavioral change is most effective for this uniquely calibrated individual."
