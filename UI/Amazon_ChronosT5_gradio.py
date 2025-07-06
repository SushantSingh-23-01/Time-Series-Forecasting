import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from chronos import ChronosPipeline
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import re
import gc
plt.style.use('dark_background')

@dataclass
class SharedState:
    df: Optional[pd.DataFrame] = None
    x_in: Optional[np.ndarray] = None
    y_in: Optional[np.ndarray] = None
    x_out: Optional[np.ndarray] = None
    y_out: Optional[np.ndarray] = None
    y_preds: Optional[np.ndarray] = None
    
    model_name: Optional[str] = None
    model_settings: Dict[str, Any] = field(default_factory=lambda: {'temperature': None, 'top_k': None, 'top_p': None})

class CSVParser:
    def _parse_csv(self, file_path):
        # read csv
        df = pd.read_csv(file_path)
        
        # clean up columns names - Remove spaces and uppercase letters
        new_cols = {}
        for col in df.columns:
            new_cols[col] = re.sub(r'\s', '', col.lower())
        df = df.rename(columns = new_cols)
        
        # Find datetime columns 
        datetime_cols_found = []
        for col in df.columns:
            if any(keyword in col for keyword in ['date', 'datetime', 'time']):
                datetime_cols_found.append(col)
        
        # Attempt to convert identified datetime columns to datetime objects
        for col in datetime_cols_found:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Set datetime columns as index
        df = df.set_index(df.columns[0])
        return df  

    def _read_csv(self, file_path, shared_state):
        current_state = shared_state
        if file_path is None:
            
            return (
                current_state,
                gr.Dropdown(choices=[], value=None, interactive=False),
                gr.Dropdown(choices=[], value=None, interactive=False),
            )
            
        current_state.df = self._parse_csv(file_path)

        return (
            current_state,
            gr.Dropdown(choices=list(current_state.df.columns), value=None, interactive=True),
            gr.Dropdown(choices=list(current_state.df.columns), value=None, interactive=True),
        )

class MinMaxScaler:
    def __init__(self, new_min = 0, new_max = 1):
        self.new_min = new_min
        self.new_max = new_max
    
    def fit(self, data):
        self.old_min = np.min(data, axis=0)
        self.old_max = np.max(data, axis=0)
        return self

    def transform(self, data):
        return (data - self.old_min) / (self.old_max - self.old_min) * (self.new_max - self.new_min) + self.new_min

    def fit_transform(self, data):
        return self.fit(data).transform(data)
    
    def inverse_transform(self, scaled_data):
        return (scaled_data - self.new_min) / (self.new_max - self.new_min) * (self.old_max - self.old_min) + self.old_min

def get_input_output_splits(x_col, y_col, shared_state, in_len = 64, out_len = 8):
    current_state = shared_state
    x = current_state.df[x_col].to_numpy()
    y = current_state.df[y_col].to_numpy()
    total_samples= len(y) - in_len - out_len

    rng = np.random.default_rng()
    random_idx = rng.choice(total_samples)
    
    current_state.x_in = x[random_idx: random_idx + in_len]
    current_state.y_in = y[random_idx: random_idx + in_len]
    current_state.x_out = x[random_idx + in_len: random_idx + in_len + out_len]
    current_state.y_out = y[random_idx + in_len: random_idx + in_len + out_len]
    return current_state

def generate_random_walk(ax, shared_state, step_size=0.1):
    scaler = MinMaxScaler()
    current_state = shared_state
    y_scaled_in = scaler.fit_transform(current_state.y_in)
    y_last = y_scaled_in[-1]
    out_len = len(shared_state.y_out)
    
    rng = np.random.default_rng()
    steps = rng.standard_normal(out_len) * step_size
    # Calculate the cumulative sum of steps (error terms)
    # y_last broadcasted and added to all cumulated error terms
    random_walk = y_last + np.cumsum(steps)
    
    random_walk = scaler.inverse_transform(random_walk)
    ax.plot(current_state.x_out, random_walk, label='random walk preds')
    ax.legend()

class ChronosPrediction:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self, model_name, shared_state):
        current_state = shared_state
        current_state.model_name = model_name
        self.pipeline = ChronosPipeline.from_pretrained(current_state.model_name, device_map = self.device)
        status = f"Model '{current_state.model_name}' loaded successfully on {self.device}."
        return current_state, status
    
    def unload_model(self, shared_state):
        current_state = shared_state
        del self.pipeline
        self.pipeline = None
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        if current_state.y_preds is not None:
            del current_state.y_preds
            current_state.y_preds = None
        
        status = f"Model '{current_state.model_name}' unloaded successfully."
        current_state.model_name = None
        return current_state, status
        
    def _predict(
        self, 
        shared_state,
        output_ctx_len=8, 
        temperature=1.0, 
        top_k=50, 
        top_p=0.9,
        ):
        current_state = shared_state
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            pass
        
        else:
            y_preds = self.pipeline.predict(
                torch.tensor(current_state.y_in),
                prediction_length=output_ctx_len, 
                temperature = temperature,
                top_k = top_k,
                top_p = top_p
            ).squeeze(0).numpy()
            current_state.y_preds = y_preds
        return current_state

    def _plot_results(self, ax, shared_state):
        current_state = shared_state
        current_state = self._predict(
            shared_state=shared_state, 
            output_ctx_len=8, 
            temperature=current_state.model_settings['temperature'], 
            top_k=current_state.model_settings['top_k'], 
            top_p=current_state.model_settings['top_p']
            )
        y_preds = current_state.y_preds

        if self.pipeline is not None:
            if y_preds.ndim > 1:
                y_preds = np.mean(y_preds, axis=0)
            ax.plot(current_state.x_out, y_preds, label='Chronos forecast')
            ax.legend()
 
class InferenceUI:
    def __init__(self):
        with gr.Row():
            with gr.Column():
                gr.Markdown('### Load Files')
                self.csv_file = gr.File(label='Drop your csv file', file_types=['.csv'], height=140)
                
                with gr.Row():
                    self.x_col = gr.Dropdown(label='X data', choices=[], value=None, interactive=False)
                    self.y_col = gr.Dropdown(label='Y data', choices=[], value=None, interactive=False)
                    
                with gr.Row():
                    self.model_name = gr.Dropdown(
                        choices=['amazon/chronos-t5-tiny', 'amazon/chronos-t5-small'],
                        value='amazon/chronos-t5-tiny',
                        label='Model',
                        interactive=True,
                    )
                    with gr.Column():
                        self.load_model_btn = gr.Button('Load Model', interactive=True)
                        self.unload_model_btn = gr.Button('Unload Model', interactive=True)
                self.model_status = gr.Textbox(label='Model Status', interactive=False)
                        
                self.plot_button = gr.Button(value='Plot Data', interactive=True)
            with gr.Column():
                gr.Markdown('### Visualization')
                self.infernce_plot = gr.Plot(visible=False)   
    
    def create_base_fig(self):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Time Series Forecast')
        ax.set_xlabel(self.x_col.value)
        ax.set_ylabel(self.y_col.value)  
        return fig, ax

class SettingsUI:
    def __init__(self):
        with gr.Row():
            with gr.Accordion(label='Model Settings', open=False):           
                with gr.Column():
                    self.temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label='Temperature',
                        interactive=True,
                    )
                    self.top_k = gr.Slider(
                        minimum=0,
                        maximum=100, 
                        step=1, 
                        value=50,
                        label='Top-k',
                        interactive=True,
                    )
                    
                    self.top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.9,
                        label='Top-p',
                        interactive=True,
                    )
                    self.set_settings = gr.Button(value='Set Settings')
                
    def change_settings(self, shared_state):
        current_state = shared_state
        current_state.model_settings = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        return current_state
        
class MainUI:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.csv_loader = CSVParser()
        self._model_pipeline = ChronosPrediction()
        self._interface()
        self._events()
        
        
    def _interface(self):
        gr.Markdown("# TimeSeries UI")
        with gr.Sidebar():
            gr.Markdown("## Navigation")
            self.infer_btn = gr.Button("Inference")
            self.settings_btn = gr.Button("Settings")
    
        with gr.Column() as main_content_area:
            with gr.Column(visible=True) as self.main_page: # Initially visible
                gr.Markdown("### Inference")
                self.inference_ui = InferenceUI()
            
            with gr.Column(visible=False) as self.settings_page: # Initially visible
                gr.Markdown("### Settings")
                self.settings_ui = SettingsUI()
           
    def get_plot(self, x_col_val, y_col_val, shared_state):
        current_state = shared_state
        
        if current_state.df is None:
            return gr.Plot(visible=False), current_state
        
        elif x_col_val is None or y_col_val is None:
            return gr.Plot(visible=False), current_state
        
        else:
            current_state = get_input_output_splits(x_col_val, y_col_val, current_state)

            
            fig, ax = self.inference_ui.create_base_fig()
            ax.plot(current_state.x_in, current_state.y_in, label='train_data')
            ax.plot(current_state.x_out, current_state.y_out, label='true preds')
            generate_random_walk(ax, current_state)
            self._model_pipeline._plot_results(ax, current_state)
            
            ax.legend()
            plt.close(fig)
            return gr.Plot(fig, visible=True), current_state

    def _sidebar_events(self):
        self.infer_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[self.main_page, self.settings_page]
        )
        self.settings_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[self.main_page, self.settings_page]
        )
         
    def _inference_tab_events(self):
        self.inference_ui.csv_file.change(
            fn=self.csv_loader._read_csv,
            inputs=[self.inference_ui.csv_file, self.shared_state],
            outputs=[self.shared_state, self.inference_ui.x_col, self.inference_ui.y_col]
        )      

        self.inference_ui.plot_button.click(
            fn=self.get_plot,
            inputs=[self.inference_ui.x_col, self.inference_ui.y_col, self.shared_state],
            outputs=[self.inference_ui.infernce_plot, self.shared_state]
        )
        
        self.inference_ui.load_model_btn.click(
            fn=self._model_pipeline.load_model,
            inputs=[self.inference_ui.model_name, self.shared_state],
            outputs=[self.shared_state, self.inference_ui.model_status]
        )
        self.inference_ui.unload_model_btn.click(
            fn=self._model_pipeline.unload_model,
            inputs=[self.shared_state],
            outputs=[self.shared_state, self.inference_ui.model_status] 
        )
        
    def _setting_tab_events(self):
        self.settings_ui.set_settings.click(
            fn=self.settings_ui.change_settings,
            inputs=[self.shared_state],
            outputs=[self.shared_state]
        )

    def _events(self):
        self._sidebar_events()
        self._inference_tab_events()
        self._setting_tab_events()
    
with gr.Blocks() as demo:
    app_shared_state = gr.State(SharedState())
    MainUI(app_shared_state)

demo.launch()
