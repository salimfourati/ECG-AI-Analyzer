import numpy as np 
import plotly.graph_objs as go
import streamlit as st

def plot_all_leads(signal_matrix, fs, lead_names, seconds=10):
    """
    Affichage interactif des leads ECG avec Plotly.
    Chaque lead est affiché décalé verticalement (comme sur une feuille ECG).
    """
    n_leads = signal_matrix.shape[1]
    samples_to_show = int(seconds * fs)
    time = np.arange(samples_to_show) / fs

    fig = go.Figure()

    spacing = 5  # Espace vertical entre les leads

    tickvals = []
    ticktext = []

    for i, lead in enumerate(lead_names):
        if i < n_leads:
            y_shift = i * spacing
            signal_shifted = signal_matrix[:samples_to_show, i] + y_shift
            fig.add_trace(go.Scatter(
                x=time,
                y=signal_shifted,
                mode='lines',
                line=dict(color='black', width=1),
                name=lead.upper(),
                hovertemplate=f"Lead {lead.upper()}<br>Temps: %{{x:.2f}}s<br>Amplitude: %{{y:.2f}} mV<extra></extra>"
            ))
            tickvals.append(y_shift)
            ticktext.append(lead.upper())

    # Mise en forme type papier millimétré
    fig.update_layout(
        height=400 + 40 * n_leads,
        title="ECG – tous les leads (interactif)",
        xaxis_title="Temps (s)",
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            tickvals=tickvals,
            ticktext=ticktext,
            title="Leads"
        ),
        plot_bgcolor="white",
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='red',
        dtick=0.2, minor=dict(dtick=0.04, showgrid=True, gridcolor='rgba(255,0,0,0.3)')
    )

    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='red',
        minor=dict(dtick=0.5, showgrid=True, gridcolor='rgba(255,0,0,0.3)')
    )

    st.plotly_chart(fig, use_container_width=True)
    
def plot_with_rpeaks(signal, rpeaks_idx, fs, seconds=10):
    import plotly.graph_objs as go
    import numpy as np
    n = min(len(signal), int(seconds*fs))
    time = np.arange(n)/fs
    sig = signal[:n]
    rp = rpeaks_idx[rpeaks_idx < n]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=sig, mode='lines', line=dict(color='black', width=1)))
    fig.add_trace(go.Scatter(x=rp/fs, y=sig[rp], mode='markers', marker=dict(size=6, color='red'), name='R'))
    fig.update_layout(height=300, title="Signal + R-peaks", xaxis_title="Temps (s)", yaxis_title="mV", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

def plot_rr_tachogram(rr_time_s, rr_ms):
    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr_time_s, y=rr_ms, mode='lines+markers'))
    fig.update_layout(height=300, title="Tachogramme (RR en ms)", xaxis_title="Temps (s)", yaxis_title="RR (ms)", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

