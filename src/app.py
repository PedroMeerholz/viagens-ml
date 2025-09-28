import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from main_analysis import executar_analise
from models_config import PARAM_GRIDS


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Seletor e Analisador de Modelos de ML")
        self.root.geometry("1200x850")
        self.log_queue = queue.Queue()
        self.best_model_path = None
        self.df_results = None

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_treinamento = ttk.Frame(self.notebook)
        self.tab_analise = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_treinamento, text="Treinamento de Modelos")
        self.notebook.add(self.tab_analise, text="Análise de Resultados")

        self.create_training_tab()
        self.create_analysis_tab()

        self.root.after(100, self.process_log_queue)

    def create_training_tab(self):
        main_frame = ttk.Frame(self.tab_treinamento, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.model_vars = {}
        self.hyper_param_var = tk.BooleanVar(value=False)
        self.cross_val_var = tk.BooleanVar(value=False)
        self.create_model_selection_frame(main_frame)
        self.create_options_frame(main_frame)
        self.create_control_frame(main_frame)
        self.create_log_frame(main_frame)

    def create_model_selection_frame(self, parent):
        models_frame = ttk.LabelFrame(parent, text="Selecione os Modelos", padding="10")
        models_frame.pack(fill=tk.X, pady=5)
        model_list = sorted(list(PARAM_GRIDS.keys()))
        num_cols = 5
        for i, model_name in enumerate(model_list):
            self.model_vars[model_name] = tk.BooleanVar()
            cb = ttk.Checkbutton(models_frame, text=model_name, variable=self.model_vars[model_name])
            cb.grid(row=i // num_cols, column=i % num_cols, sticky=tk.W, padx=5, pady=2)

    def create_options_frame(self, parent):
        options_frame = ttk.LabelFrame(parent, text="Opções de Treinamento", padding="10")
        options_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(options_frame, text="Realizar Hiperparametrização (GridSearch)",
                        variable=self.hyper_param_var).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(options_frame, text="Usar Validação Cruzada",
                        variable=self.cross_val_var).pack(side=tk.LEFT, padx=10)

    def create_log_frame(self, parent):
        log_frame = ttk.LabelFrame(parent, text="Logs da Execução", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state='disabled', height=10)
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def create_control_frame(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Controles e Progresso", padding="10")
        control_frame.pack(fill=tk.X, pady=10)
        button_container = ttk.Frame(control_frame)
        button_container.pack(fill=tk.X, pady=(0, 5))
        self.run_button = ttk.Button(button_container, text="Iniciar Análise", command=self.run_analysis_thread)
        self.run_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.copy_button = ttk.Button(button_container, text="Copiar Log", command=self.copy_log_to_clipboard)
        self.copy_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.export_button = ttk.Button(button_container, text="Exportar Melhor Modelo (.pkl)",
                                        command=self.export_best_model, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.progress_bar = ttk.Progressbar(control_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, expand=True)

    def create_analysis_tab(self):
        analysis_frame = ttk.Frame(self.tab_analise, padding="10")
        analysis_frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(analysis_frame)
        controls.pack(fill=tk.X, pady=5)

        ttk.Button(controls, text="Carregar Resultados da Última Análise", command=self.load_results_data).pack(
            side=tk.LEFT, padx=5)

        self.chart_type = tk.StringVar(value="bar")
        ttk.Label(controls, text="Tipo de Gráfico:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Radiobutton(controls, text="Barras", variable=self.chart_type, value="bar",
                        command=self.update_analysis_chart).pack(side=tk.LEFT)
        ttk.Radiobutton(controls, text="Pontos/Linhas", variable=self.chart_type, value="line",
                        command=self.update_analysis_chart).pack(side=tk.LEFT)

        self.figure_frame = ttk.Frame(analysis_frame)
        self.figure_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_results_data(self):
        results_path = "artifacts/resultados_completos.csv"
        if not os.path.exists(results_path):
            messagebox.showwarning("Aviso", "Arquivo de resultados não encontrado. Execute uma análise primeiro.")
            self.df_results = None
            return

        self.df_results = pd.read_csv(results_path)
        messagebox.showinfo("Sucesso", "Resultados carregados. Selecione um tipo de gráfico para visualizar.")
        self.update_analysis_chart()

    def update_analysis_chart(self):
        if self.df_results is None:
            messagebox.showwarning("Aviso", "Nenhum resultado carregado. Clique em 'Carregar Resultados' primeiro.")
            return

        self.fig.clear()

        df = self.df_results.copy()
        df.rename(columns={'acuracia': 'Acurácia', 'f1_score': 'F1-Score',
                           'precisao': 'Precisão', 'recall': 'Recall'}, inplace=True)

        df_melted = df.melt(id_vars='modelo', var_name='Métrica', value_name='Valor')

        # --- Alteração aqui: Ordenar pela Acurácia ---
        order = df.sort_values('Acurácia', ascending=False).modelo

        ax = self.fig.add_subplot(111)

        chart_type = self.chart_type.get()
        if chart_type == "bar":
            sns.barplot(data=df_melted, y='modelo', x='Valor', hue='Métrica', ax=ax, order=order)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)
        else:
            sns.pointplot(data=df_melted, y='modelo', x='Valor', hue='Métrica', ax=ax, order=order, dodge=True)
            ax.grid(True, linestyle='--', alpha=0.6)

        ax.set_title('Comparação de Métricas dos Modelos', fontsize=16)
        ax.set_ylabel('Modelo', fontsize=12)
        ax.set_xlabel('Pontuação', fontsize=12)
        ax.set_xlim(0, max(1.0, df_melted['Valor'].max() * 1.1))
        ax.legend(title='Métrica')

        self.fig.tight_layout()
        self.canvas.draw()

    def copy_log_to_clipboard(self):
        log_content = self.log_area.get('1.0', tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(log_content)
        messagebox.showinfo("Copiado", "O conteúdo do log foi copiado para a área de transferência.")

    def log_message(self, message):
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, message + '\n')
        self.log_area.configure(state='disabled')
        self.log_area.see(tk.END)

    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                if isinstance(message, tuple):
                    if message[0] == 'progress':
                        self.progress_bar['value'] = message[1]
                    elif message[0] == 'best_model_path':
                        self.best_model_path = message[1]
                        if self.best_model_path:
                            self.export_button.config(state=tk.NORMAL)
                else:
                    self.log_message(str(message))
        except queue.Empty:
            pass
        self.root.after(100, self.process_log_queue)

    def run_analysis_thread(self):
        modelos_selecionados = [name for name, var in self.model_vars.items() if var.get()]
        if not modelos_selecionados:
            messagebox.showerror("Erro", "Nenhum modelo foi selecionado.")
            return

        self.run_button.config(state=tk.DISABLED)
        self.copy_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)
        self.log_area.configure(state='normal')
        self.log_area.delete('1.0', tk.END)
        self.log_area.configure(state='disabled')
        self.best_model_path = None

        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = len(modelos_selecionados)

        analysis_thread = threading.Thread(
            target=executar_analise,
            args=(modelos_selecionados, self.hyper_param_var.get(), self.cross_val_var.get(), self.log_queue),
            daemon=True
        )
        analysis_thread.name = 'AnalysisThread'
        analysis_thread.start()
        self.root.after(200, self.check_thread)

    def check_thread(self):
        is_running = any(t.is_alive() for t in threading.enumerate() if t.name == 'AnalysisThread')
        if is_running:
            self.root.after(200, self.check_thread)
        else:
            self.run_button.config(state=tk.NORMAL)
            self.copy_button.config(state=tk.NORMAL)

    def export_best_model(self):
        if not self.best_model_path or not os.path.exists(self.best_model_path):
            messagebox.showerror("Erro", "Nenhum modelo para exportar ou arquivo não encontrado.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")],
            title="Salvar o melhor modelo como..."
        )
        if file_path:
            try:
                shutil.copy(self.best_model_path, file_path)
                messagebox.showinfo("Sucesso", f"Modelo salvo com sucesso em:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Erro na Exportação", f"Não foi possível salvar o arquivo:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()