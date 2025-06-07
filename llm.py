from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
login(token="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")

# Încărcăm modelul și tokenizer-ul
model_name = "tiiuae/falcon-7b"  # Poți înlocui cu alt model dacă dorești
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Verificăm disponibilitatea GPU-ului
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Inițializăm istoricul conversației și regulile
conversation_history = []
rules = []

# Funcție pentru construirea prompt-ului de răspuns
def build_answer_prompt(user_input, rules, history):
    prompt = ""
    if rules:
        prompt += "Reguli curente:\n" + "\n".join(f"- {rule}" for rule in rules) + "\n\n"
    if history:
        prompt += "Istoric conversație:\n" + "\n".join(f"Utilizator: {q}\nAsistent: {a}" for q, a in history) + "\n\n"
    prompt += f"Utilizator: {user_input}\nAsistent:"
    return prompt

# Funcție pentru construirea prompt-ului de propunere a unei noi reguli
def build_rule_prompt(rules, history):
    prompt = "Pe baza următoarei conversații și a regulilor existente, propune o nouă regulă sau idee care ar putea fi aplicată în interacțiunile viitoare.\n\n"
    if rules:
        prompt += "Reguli existente:\n" + "\n".join(f"- {rule}" for rule in rules) + "\n\n"
    if history:
        prompt += "Conversație:\n" + "\n".join(f"Utilizator: {q}\nAsistent: {a}" for q, a in history) + "\n\n"
    prompt += "Regulă nouă propusă:"
    return prompt

# Buclă interactivă
while True:
    user_input = input("Tu: ").strip()
    
    if user_input.lower() == "exit":
        print("Ieșire...")
        break
    elif user_input.lower().startswith("adaugă regulă:"):
        new_rule = user_input[len("adaugă regulă:"):].strip()
        rules.append(new_rule)
        print(f"Regulă adăugată: {new_rule}")
    elif user_input.lower().startswith("elimină regulă:"):
        try:
            index = int(user_input[len("elimină regulă:"):].strip())
            if 0 <= index < len(rules):
                removed_rule = rules.pop(index)
                print(f"Regulă eliminată: {removed_rule}")
            else:
                print("Indice regulă invalid.")
        except ValueError:
            print("Te rog să furnizezi un indice valid.")
    elif user_input.lower() == "arată reguli":
        if rules:
            print("Reguli curente:")
            for i, rule in enumerate(rules):
                print(f"{i}: {rule}")
        else:
            print("Nu există reguli setate momentan.")
    else:
        # Construim prompt-ul pentru răspuns
        answer_prompt = build_answer_prompt(user_input, rules, conversation_history)
        inputs = tokenizer(answer_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)[len(answer_prompt):].strip()
        print("Asistent:", answer)
        
        # Adăugăm la istoricul conversației
        conversation_history.append((user_input, answer))
        
        # Construim prompt-ul pentru propunerea unei noi reguli
        rule_prompt = build_rule_prompt(rules, conversation_history)
        rule_inputs = tokenizer(rule_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            rule_output = model.generate(**rule_inputs, max_new_tokens=50, temperature=0.7)
        proposed_rule = tokenizer.decode(rule_output[0], skip_special_tokens=True)[len(rule_prompt):].strip()
        print("Regulă nouă propusă:", proposed_rule)
        
        # Întrebăm utilizatorul dacă dorește să adauge regula
        accept = input("Adaugi această regulă? (da/nu): ").lower()
        if accept == "da":
            rules.append(proposed_rule)
            print("Regulă adăugată.")
        else:
            print("Regulă respinsă.")