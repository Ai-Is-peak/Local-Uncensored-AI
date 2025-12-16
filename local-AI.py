import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_HISTORY_MESSAGES = 10

#Token limit change how you want, kinda low token limit but after all this is made to be able to run on budget devices without gpus and around 8gb of ram.
VERY_LARGE_MAX_NEW_TOKENS = 4096

MODELS = {
    "TinyLlama Chat": {
        "folder": "TinyLlama-1.1B-Chat-v1.0",
        "mode": "chat",
        "description": "OK kinda dumb assistant chat AI",
    },
    "TinyLlama Math": {
        "folder": "TinyLlama-Math",
        "mode": "math",
        "description": "Dumb Math assistant",
    },
    "distilgpt2": {
        "folder": "distilgpt2",
        "mode": "gpt2",
        "description": "Small text generator, very dumb and types random stuff.",
    },
}

STOP_REQUESTED = False


def system_prompt_for_mode(mode: str) -> str:
    base_rules = (
        "Rules:\n"
        "- Only write the assistant's reply.\n"
        "- Never write 'User:' / 'Human:' / 'SYSTEM:' or pretend to be the user.\n"
        "- Never continue the conversation as multiple turns. Write only ONE assistant message.\n"
        "- Do NOT create dialogue. Do NOT add new user messages.\n"
        "- YOUR name is SondreGPT.\n"
        "- If you accidentally start writing 'User:' or similar, stop and continue as assistant.\n"
    )

    if mode == "math":
        return (
            "You are a careful math tutor.\n"
            + base_rules
            + "Explain clearly with as many steps as needed, then give the final answer.\n"
        )

    return "You are a helpful, friendly assistant.\n" + base_rules


def build_prompt_from_history(history, mode: str) -> str:

    history = history[-MAX_HISTORY_MESSAGES:]

    if mode in ("chat", "math"):
        system = system_prompt_for_mode(mode)
        parts = [f"<|system|>\n{system}\n"]

        for msg in history:
            if msg["role"] == "user":
                parts.append(f"<|user|>\n{msg['content']}\n")
            elif msg["role"] == "assistant":
                parts.append(f"<|assistant|>\n{msg['content']}\n")

        parts.append("<|assistant|>\n")
        return "".join(parts)

    lines = []
    for msg in history:
        prefix = "User" if msg["role"] == "user" else "AI"
        lines.append(f"{prefix}: {msg['content']}")
    lines.append("AI:")
    return "\n".join(lines) + "\n"


def postprocess_reply(text: str) -> str:
    return text.strip()


def _make_bad_words_ids(tokenizer):
    phrases = [
        "User:", "USER:", "Human:", "HUMAN:",
        "Assistant:", "ASSISTANT:", "System:", "SYSTEM:",
        "<|user|>", "<|assistant|>", "<|system|>",
    ]
    bad = []
    for p in phrases:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if ids:
            bad.append(ids)
    return bad


def generate_reply(history, tokenizer, model, mode: str) -> str:
    global STOP_REQUESTED
    STOP_REQUESTED = False

    prompt = build_prompt_from_history(history, mode)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    if mode == "math":
        temperature = 0.35
        top_p = 0.9
    else:
        temperature = 0.6
        top_p = 0.9

    bad_words_ids = None
    if mode in ("chat", "math"):
        bad_words_ids = _make_bad_words_ids(tokenizer)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=VERY_LARGE_MAX_NEW_TOKENS,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id or None,
            bad_words_ids=bad_words_ids,
        )

    if mode in ("chat", "math"):
        new_tokens = output_ids[0, input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return postprocess_reply(text)


class LocalAIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Local AI | SondreGPT")

        self.tokenizer = None
        self.model = None
        self.current_mode = None
        self.current_model_name = None

        self.is_loading = False
        self.is_generating = False
        self.think_phase = 0

        self.history = []
        self.chat_saved = True

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        main.columnconfigure(0, weight=1)

        top = ttk.Frame(main)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Model:").grid(row=0, column=0, sticky="w")

        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(
            top, textvariable=self.model_var, state="readonly", values=list(MODELS.keys())
        )
        if MODELS:
            self.model_box.current(0)
        self.model_box.grid(row=0, column=1, sticky="ew", padx=5)

        self.load_button = ttk.Button(top, text="Load model", command=self.on_load_model)
        self.load_button.grid(row=0, column=2, padx=(0, 5))

        self.save_button = ttk.Button(top, text="Save chat", command=self.on_save_chat)
        self.save_button.grid(row=0, column=3, padx=(0, 5))

        self.load_chat_button = ttk.Button(top, text="Load chat", command=self.on_load_chat)
        self.load_chat_button.grid(row=0, column=4, padx=(0, 5))

        # NEW: Stop button (important when output is long)
        self.stop_button = ttk.Button(top, text="Stop", command=self.on_stop)
        self.stop_button.grid(row=0, column=5)

        self.status_var = tk.StringVar(value="Select a model and click 'Load model'.")
        ttk.Label(top, textvariable=self.status_var).grid(
            row=1, column=0, columnspan=6, sticky="w", pady=(3, 0)
        )

        self.chat = tk.Text(main, wrap="word", height=18, state="disabled", bd=0, padx=5, pady=5)
        self.chat.grid(row=2, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(main, command=self.chat.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.chat["yscrollcommand"] = scroll.set

        self.chat.tag_configure(
            "user",
            background="#3b82f6",
            foreground="white",
            lmargin1=80,
            lmargin2=80,
            rmargin=10,
            spacing3=5,
            spacing1=5,
            spacing2=2,
        )
        self.chat.tag_configure(
            "ai",
            background="#e5e7eb",
            foreground="#111827",
            lmargin1=10,
            lmargin2=10,
            rmargin=80,
            spacing3=5,
            spacing1=5,
            spacing2=2,
        )
        self.chat.tag_configure(
            "system",
            background="#f3f4f6",
            foreground="#4b5563",
            lmargin1=40,
            lmargin2=40,
            rmargin=40,
            spacing3=5,
            font=("TkDefaultFont", 9, "italic"),
        )

        bottom = ttk.Frame(main)
        bottom.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        bottom.columnconfigure(0, weight=1)

        self.input_var = tk.StringVar()
        self.entry = ttk.Entry(bottom, textvariable=self.input_var)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<Return>", self.on_send)

        self.send_button = ttk.Button(bottom, text="Send", command=self.on_send)
        self.send_button.grid(row=0, column=1, padx=(5, 0))

    def on_stop(self):
        messagebox.showinfo(
            "Stop",
            "This model generates everythin in one call. Stop mid answer\n"
            "If it is slow, wait or restart the application\n"
        )

    def set_status(self, msg: str):
        self.status_var.set(msg)

    def set_buttons_enabled(self, enabled: bool):
        s = "normal" if enabled else "disabled"
        self.load_button["state"] = s
        self.send_button["state"] = s
        self.save_button["state"] = s
        self.load_chat_button["state"] = s
        self.stop_button["state"] = s
        self.entry["state"] = s if enabled else "disabled"

    def chat_add(self, who: str, text: str, tag: str):
        self.chat.config(state="normal")
        self.chat.insert("end", f"{who}: {text}\n", tag)
        self.chat.insert("end", "\n")
        self.chat.see("end")
        self.chat.config(state="disabled")

        if tag in ("user", "ai"):
            self.chat_saved = False

    def clear_chat_view(self):
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")

    def start_thinking_animation(self):
        self.think_phase = 0
        self._animate_thinking()

    def _animate_thinking(self):
        if not self.is_generating:
            return
        dots = "." * (self.think_phase % 4)
        self.set_status(f"Thinking{dots}")
        self.think_phase += 1
        self.root.after(400, self._animate_thinking)

    def on_load_model(self):
        if self.is_loading:
            return

        name = self.model_var.get()
        if not name:
            messagebox.showwarning("No model", "Please select a model.")
            return

        cfg = MODELS[name]
        folder = cfg["folder"]

        self.is_loading = True
        self.set_buttons_enabled(False)
        self.set_status(f"Loading {name} ...")

        def worker():
            try:
                model_path = os.path.join(SCRIPT_DIR, folder)
                tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                mdl = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
                mdl.eval()

                def finish():
                    self.tokenizer = tok
                    self.model = mdl
                    self.current_mode = cfg["mode"]
                    self.current_model_name = name

                    self.history = []
                    self.chat_saved = True
                    self.clear_chat_view()

                    self.chat_add("SYSTEM", f"Loaded {name}", "system")
                    self.set_status(f"{name} loaded. {cfg['description']}")
                    self.set_buttons_enabled(True)
                    self.entry.focus()
                    self.is_loading = False

                self.root.after(0, finish)

            except Exception as e:
                def on_err():
                    self.set_status("Failed to load model.")
                    messagebox.showerror("Load error", str(e))
                    self.set_buttons_enabled(True)
                    self.is_loading = False

                self.root.after(0, on_err)

        threading.Thread(target=worker, daemon=True).start()

    def on_send(self, event=None):
        if self.is_loading:
            messagebox.showinfo("Please wait", "Model is still loading.")
            return
        if self.model is None or self.tokenizer is None:
            messagebox.showwarning("No model", "Load a model first.")
            return
        if self.is_generating:
            return

        text = self.input_var.get().strip()
        if not text:
            return

        self.input_var.set("")
        self.chat_add("You", text, "user")
        self.history.append({"role": "user", "content": text})

        self.is_generating = True
        self.set_buttons_enabled(False)
        self.start_thinking_animation()

        mode = self.current_mode
        tokenizer = self.tokenizer
        model = self.model
        history_snapshot = list(self.history)

        def worker():
            try:
                reply = generate_reply(history_snapshot, tokenizer, model, mode)

                def finish():
                    self.history.append({"role": "assistant", "content": reply})
                    self.chat_add("AI", reply, "ai")
                    self.is_generating = False
                    self.set_status("Ready.")
                    self.set_buttons_enabled(True)
                    self.entry.focus()

                self.root.after(0, finish)

            except Exception as e:
                def on_err():
                    self.is_generating = False
                    self.set_status("Error during generation.")
                    messagebox.showerror("Generation error", str(e))
                    self.set_buttons_enabled(True)

                self.root.after(0, on_err)

        threading.Thread(target=worker, daemon=True).start()

    def save_chat_to_json(self) -> bool:
        if not self.history:
            messagebox.showinfo("Nothing to save", "No chat history to save.")
            return False

        default_name = "chat_history.json"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=default_name,
            title="Save chat history as JSON",
        )
        if not filepath:
            return False

        payload = {
            "model": self.current_model_name,
            "mode": self.current_mode,
            "history": self.history,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.chat_saved = True
            self.set_status(f"Chat saved to {os.path.basename(filepath)}")
            return True
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return False

    def on_save_chat(self):
        self.save_chat_to_json()

    def on_load_chat(self):
        if self.history and not self.chat_saved:
            if messagebox.askyesno("Save chat", "Chat is not saved. Save as JSON?"):
                if not self.save_chat_to_json():
                    return

        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load chat history (JSON)",
        )
        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if isinstance(payload, dict) and "history" in payload:
                history = payload["history"]
            elif isinstance(payload, list):
                history = payload
            else:
                raise ValueError("Invalid JSON format. Expected a list or a dict with 'history'.")

            if not isinstance(history, list):
                raise ValueError("Invalid history format inside JSON.")

            self.history = []
            self.clear_chat_view()

            self.chat_add("SYSTEM", f"Loaded chat from {os.path.basename(filepath)}", "system")

            for msg in history:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    continue
                role = msg["role"]
                content = str(msg["content"])
                if role == "user":
                    self.chat_add("You", content, "user")
                    self.history.append({"role": "user", "content": content})
                elif role == "assistant":
                    self.chat_add("AI", content, "ai")
                    self.history.append({"role": "assistant", "content": content})

            self.chat_saved = True
            self.set_status("Chat loaded.")

        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def on_close(self):
        if self.history and not self.chat_saved:
            if messagebox.askyesno("Save chat", "Chat is not saved. Save as JSON?"):
                if not self.save_chat_to_json():
                    return
        self.root.destroy()


def main():
    root = tk.Tk()
    app = LocalAIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
