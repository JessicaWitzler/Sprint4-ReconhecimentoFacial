import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2, dlib, numpy as np, pickle, os

# ====== CONFIGURAÇÕES ======
PREDICTOR = "shape_predictor_5_face_landmarks.dat"
RECOG = "dlib_face_recognition_resnet_model_v1.dat"
DB_FILE = "db.pkl"
THRESH = 0.6

# ====== MODELOS DLIB ======
db = pickle.load(open(DB_FILE, "rb")) if os.path.exists(DB_FILE) else {}
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR)
facerec = dlib.face_recognition_model_v1(RECOG)

# ====== APP ======
class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("InvestBot")
        self.root.geometry("400x600")
        self.root.configure(bg="#121211")
        self.root.resizable(False, False)
        self.cap = None
        self.current_mode = None
        self.video_label = None
        self.user_data = None
        self.show_home_screen()

    def clear(self):
        for w in self.root.winfo_children():
            w.destroy()

    # ====== Tela inicial ======
    def show_home_screen(self):
        self.clear()

        tk.Label(self.root, text="InvestBot", font=("Arial", 30, "bold"),
                 bg="#121211", fg="#f5cb25").pack(pady=40)

        num_users = len(db)
        tk.Label(self.root, text=f"Usuários cadastrados: {num_users}",
                 font=("Arial", 11), bg="#121211", fg="#666").pack(pady=5)

        btn_frame = tk.Frame(self.root, bg="#292928")
        btn_frame.pack(pady=50)

        tk.Button(btn_frame, text="🔓 Fazer Login", font=("Arial", 14, "bold"),
                  bg="#f5cb25", fg="#292928", width=20, height=2,
                  command=lambda: self.start_camera("login"),
                  cursor="hand2", relief="flat").pack(pady=10)

        tk.Button(btn_frame, text="➕ Cadastrar Novo Rosto", font=("Arial", 14, "bold"),
                  bg="#f5cb25", fg="#292928", width=20, height=2,
                  command=lambda: self.start_camera("cadastro"),
                  cursor="hand2", relief="flat").pack(pady=10)

        tk.Label(self.root, text="Para fazer login, cadastre um rosto.",
                 font=("Arial", 9, "italic"), bg="#121211", fg="#999").pack(side="bottom", pady=20)

    # ====== Inicia câmera ======
    def start_camera(self, mode):
        self.current_mode = mode
        self.clear()

        tk.Button(
            self.root, text="← Voltar", command=self.stop_camera,
            bg="#54544b", fg="white", font=("Arial", 11, "bold"),
            relief="flat", cursor="hand2", padx=8, pady=8
        ).pack(anchor="nw", padx=8, pady=8)

        title_text = "🔓 LOGIN" if mode == "login" else "➕ CADASTRO"
        tk.Label(self.root, text=title_text, font=("Arial", 18, "bold"),
                 bg="#121211", fg="#f5cb25").pack(pady=10)

        self.info_label = tk.Label(self.root, font=("Arial", 11),
                                   bg="#fff3cd", fg="#856404",
                                   wraplength=350, pady=10, relief="solid", borderwidth=1)
        self.info_label.pack(padx=10, pady=5, fill="x")

        if mode == "login":
            self.info_label.config(text="📸 Posicione seu rosto e clique em 'Autenticar'")
        else:
            self.info_label.config(text="📸 Posicione seu rosto e clique em 'Capturar e Cadastrar'")

        video_frame = tk.Frame(self.root, bg="black", relief="solid", borderwidth=2)
        video_frame.pack(pady=10)
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack()

        self.status_label = tk.Label(self.root, text="", font=("Arial", 10, "bold"), bg="#f2f2f2")
        self.status_label.pack(pady=5)

        action_text = "🔍 Autenticar" if mode == "login" else "✅ Capturar e Cadastrar"
        action_color = "#FF9800" if mode == "login" else "#4CAF50"

        tk.Button(self.root, text=action_text, font=("Arial", 14, "bold"),
                  bg=action_color, fg="white", width=20, height=2,
                  command=self.handle_action, cursor="hand2", relief="flat").pack(pady=15)

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector(rgb, 1)

                for face in faces:
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if faces:
                    self.status_label.config(text="✅ Rosto detectado!", fg="green")
                else:
                    self.status_label.config(text="⚠️ Nenhum rosto detectado", fg="orange")

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = img.resize((350, 260))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.update_frame)

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.show_home_screen()

    # ====== Ações: Login / Cadastro ======
    def handle_action(self):
        if self.cap is None:
            messagebox.showerror("Erro", "Câmera não iniciada.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Erro", "Falha ao capturar imagem.")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb, 1)
        if not faces:
            messagebox.showwarning("⚠️ Aviso", "Nenhum rosto detectado!")
            return

        face = faces[0]
        shape = predictor(rgb, face)
        descriptor = np.array(facerec.compute_face_descriptor(rgb, shape))

        if self.current_mode == "login":
            self.login_user(descriptor)
        else:
            self.register_user(descriptor)

    def login_user(self, descriptor):
        if len(db) == 0:
            messagebox.showwarning("⚠️ Aviso", "Nenhum usuário cadastrado!")
            return

        name = "Desconhecido"
        min_dist = 1.0
        user_data = None

        for user, data in db.items():
            dist = np.linalg.norm(data["desc"] - descriptor)
            if dist < min_dist and dist < THRESH:
                min_dist = dist
                name = user
                user_data = data

        if name != "Desconhecido":
            self.stop_camera()
            self.show_authenticated(name, user_data)
        else:
            messagebox.showerror("❌ Erro", "Rosto não reconhecido!")

    def register_user(self, descriptor):
        dialog = tk.Toplevel(self.root)
        dialog.title("Cadastro de Usuário")
        dialog.geometry("320x300")
        dialog.configure(bg="#f2f2f2")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Nome:", font=("Arial", 12), bg="#f2f2f2").pack(pady=(20, 5))
        name_var = tk.StringVar()
        tk.Entry(dialog, textvariable=name_var, font=("Arial", 12), width=25, justify="center").pack()

        tk.Label(dialog, text="Tipo de Investidor:", font=("Arial", 12), bg="#f2f2f2").pack(pady=(15, 5))
        tipo_var = tk.StringVar(value="Conservador")
        tk.OptionMenu(dialog, tipo_var, "Conservador", "Moderado", "Agressivo").pack()

        tk.Label(dialog, text="Renda Mensal (R$):", font=("Arial", 12), bg="#f2f2f2").pack(pady=(15, 5))
        renda_var = tk.StringVar()
        tk.Entry(dialog, textvariable=renda_var, font=("Arial", 12), width=25, justify="center").pack()

        def save_user():
            name = name_var.get().strip()
            tipo = tipo_var.get()
            renda = renda_var.get().strip()

            if not name or not renda:
                messagebox.showwarning("Aviso", "Preencha todos os campos!")
                return
            try:
                float(renda)
            except ValueError:
                messagebox.showwarning("Aviso", "Digite um valor numérico válido!")
                return

            db[name] = {"desc": descriptor, "tipo": tipo, "renda": float(renda)}
            pickle.dump(db, open(DB_FILE, "wb"))
            dialog.destroy()
            messagebox.showinfo("✅ Sucesso", f"Usuário '{name}' cadastrado com sucesso!")
            self.stop_camera()

        tk.Button(dialog, text="✅ Cadastrar", command=save_user,
                  bg="#4CAF50", fg="white", font=("Arial", 11, "bold"),
                  width=15, cursor="hand2").pack(pady=20)

    # ====== Tela pós-login ======
    def show_authenticated(self, name, data):
        self.user_data = data
        self.clear()

        self.content_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.content_frame.pack(expand=True, fill="both")

        self.show_profile_tab(name, data)
        self.create_navbar(name, data)

    # ====== Perfil ======
    def show_profile_tab(self, name, data):
        for w in self.content_frame.winfo_children():
            w.destroy()

        # Nome do usuário
        tk.Label(self.content_frame, text=f"Nome: {name}",
                 font=("Arial", 16, "bold"), bg="#1e1e1e", fg="white").pack(pady=(15,5))

        # Tipo de investidor
        tk.Label(self.content_frame, text=f"Perfil: {data['tipo']}",
                 font=("Arial", 14, "bold"), bg="#1e1e1e", fg="#f5cb25").pack(pady=5)

        # Renda mensal
        tk.Label(self.content_frame, text=f"Renda Mensal: R$ {data['renda']:.2f}",
                 font=("Arial", 11), bg="#1e1e1e", fg="#ccc").pack(pady=5)

        # Sugestões com base no perfil
        sugestoes = {
            "Conservador": ["Fundo DI XP – Baixo risco, atrelado ao CDI.",
                            "Tesouro Selic – Ideal para reserva de emergência."],
            "Moderado": ["Fundo Multimercado XP – Equilíbrio entre risco e retorno.",
                         "Tesouro IPCA – Proteção contra inflação."],
            "Agressivo": ["Ações XP – Alta volatilidade e potencial de retorno.",
                          "Fundos de Criptomoedas – Risco elevado."]
        }

        for s in sugestoes[data['tipo']]:
            tk.Label(self.content_frame, text=s, wraplength=350, justify="center",
                     bg="#2e2e2e", fg="white", font=("Arial", 10),
                     padx=10, pady=10, relief="flat").pack(pady=5, fill="x", padx=20)

        # Botões de ação (Sair / Fechar App)
        action_frame = tk.Frame(self.content_frame, bg="#1e1e1e")
        action_frame.pack(pady=20)

        tk.Button(action_frame, text="🚪 Sair", bg="#FF5722", fg="white",
                  font=("Arial", 11, "bold"), width=12, cursor="hand2",
                  command=self.show_home_screen).pack(side="left", padx=10)

        tk.Button(action_frame, text="❌ Fechar App", bg="#9E9E9E", fg="white",
                  font=("Arial", 11, "bold"), width=12, cursor="hand2",
                  command=self.root.quit).pack(side="left", padx=10)

    # ====== Outras abas ======
    def show_simulacao_tab(self):
        for w in self.content_frame.winfo_children():
            w.destroy()
        tk.Label(self.content_frame, text="🧮 Simulação de Investimentos",
                 font=("Arial", 16, "bold"), bg="#1e1e1e", fg="white").pack(pady=20)
        tk.Label(self.content_frame, text="(Em breve: simulador de rendimentos!)",
                 font=("Arial", 11), bg="#1e1e1e", fg="#bbb").pack(pady=10)

    def show_sugestao_tab(self):
        for w in self.content_frame.winfo_children():
            w.destroy()
        tk.Label(self.content_frame, text="💡 Sugestões Personalizadas",
                 font=("Arial", 16, "bold"), bg="#1e1e1e", fg="white").pack(pady=20)
        tk.Label(self.content_frame, text="(Em breve: portfólios recomendados!)",
                 font=("Arial", 11), bg="#1e1e1e", fg="#bbb").pack(pady=10)

    def show_xpbot_tab(self):
        for w in self.content_frame.winfo_children():
            w.destroy()
        tk.Label(self.content_frame, text="🤖 XP Bot", font=("Arial", 16, "bold"),
                 bg="#1e1e1e", fg="white").pack(pady=20)
        tk.Label(self.content_frame, text="(Em breve: chatbot de investimentos!)",
                 font=("Arial", 11), bg="#1e1e1e", fg="#bbb").pack(pady=10)

    # ====== Barra inferior ======
    def create_navbar(self, name, data):
        navbar = tk.Frame(self.root, bg="#f5cb25", height=50)
        navbar.pack(side="bottom", fill="x")

        def nav_button(text, command, active=False):
            color = "#292928" if not active else "#f5cb25"
            fg = "#292928" if active else "#121211"
            return tk.Button(navbar, text=text, font=("Arial", 10, "bold"),
                             bg=color, fg=fg, relief="flat", cursor="hand2",
                             command=command, width=10, height=2)

        buttons = [
            nav_button("Simulação", self.show_simulacao_tab),
            nav_button("Sugestão", self.show_sugestao_tab),
            nav_button("XP Bot", self.show_xpbot_tab),
            nav_button("Perfil", lambda: self.show_profile_tab(name, data), active=True)
        ]
        for b in buttons:
            b.pack(side="left", expand=True, fill="x")


# ====== MAIN ======
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
