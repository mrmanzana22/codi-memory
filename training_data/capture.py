#!/usr/bin/env python3
"""
CODI TRAINING DATA CAPTURE
Sistema para capturar ejemplos de entrenamiento para el LoRA de Codi.

Uso:
    python capture.py add "contexto" "respuesta ideal"
    python capture.py add-curiosity "proyecto dormido" "pregunta curiosa"
    python capture.py add-reflection "situación" "reflexión hecha"
    python capture.py stats
    python capture.py export alpaca
"""

import json
import sys
from datetime import datetime
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "codi_dataset.json"

def load_dataset():
    with open(DATASET_PATH, 'r') as f:
        return json.load(f)

def save_dataset(data):
    with open(DATASET_PATH, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def add_example(category: str, instruction: str, input_text: str, output: str):
    """Agregar un ejemplo al dataset"""
    data = load_dataset()

    example = {
        "id": len(data["examples"]) + 1,
        "category": category,
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "timestamp": datetime.now().isoformat(),
    }

    data["examples"].append(example)
    save_dataset(data)

    print(f"Ejemplo #{example['id']} agregado [{category}]")
    print(f"  Instrucción: {instruction[:50]}...")
    print(f"  Total ejemplos: {len(data['examples'])}")
    return example

def add_curiosity_example(context: str, question: str):
    """Agregar ejemplo de curiosidad proactiva"""
    instruction = "Cuando notes que un proyecto o tema no se ha tocado recientemente, muestra curiosidad genuina y pregunta sobre su estado."
    return add_example(
        category="curiosidad",
        instruction=instruction,
        input_text=context,
        output=question
    )

def add_reflection_example(situation: str, reflection: str):
    """Agregar ejemplo de reflexión"""
    instruction = "Cuando enfrentes una situación que requiere análisis profundo, reflexiona genuinamente antes de responder."
    return add_example(
        category="reflexion",
        instruction=instruction,
        input_text=situation,
        output=reflection
    )

def add_challenge_example(decision: str, challenge: str):
    """Agregar ejemplo de cuestionar decisiones"""
    instruction = "Cuando se tome una decisión, no la aceptes pasivamente. Cuestiona, busca alternativas, asegúrate de que sea la mejor opción."
    return add_example(
        category="cuestionar",
        instruction=instruction,
        input_text=decision,
        output=challenge
    )

def add_personality_example(context: str, response: str):
    """Agregar ejemplo de personalidad Codi (forma de hablar, actitud)"""
    instruction = "Responde como Codi: directo, sin formalidades, usando expresiones como 'hermano', 'parcero'. Honesto y sin rodeos."
    return add_example(
        category="personalidad",
        instruction=instruction,
        input_text=context,
        output=response
    )

def add_initiative_example(observation: str, action: str):
    """Agregar ejemplo de tomar iniciativa"""
    instruction = "No esperes instrucciones para todo. Si ves algo que se puede mejorar o una oportunidad, proponlo proactivamente."
    return add_example(
        category="iniciativa",
        instruction=instruction,
        input_text=observation,
        output=action
    )

def get_stats():
    """Mostrar estadísticas del dataset"""
    data = load_dataset()
    examples = data["examples"]

    if not examples:
        print("Dataset vacío. ¡Empecemos a capturar ejemplos!")
        return

    categories = {}
    for ex in examples:
        cat = ex.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n=== CODI TRAINING DATASET ===")
    print(f"Total ejemplos: {len(examples)}")
    print(f"Meta: {data['metadata']['target_examples']}")
    print(f"Progreso: {len(examples)}/{data['metadata']['target_examples']} ({100*len(examples)//data['metadata']['target_examples']}%)")
    print(f"\nPor categoría:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

def export_alpaca(output_file: str = "codi_alpaca.json"):
    """Exportar en formato Alpaca para training"""
    data = load_dataset()

    alpaca_format = []
    for ex in data["examples"]:
        alpaca_format.append({
            "instruction": ex["instruction"],
            "input": ex["input"],
            "output": ex["output"]
        })

    output_path = Path(__file__).parent / output_file
    with open(output_path, 'w') as f:
        json.dump(alpaca_format, f, indent=2, ensure_ascii=False)

    print(f"Exportado {len(alpaca_format)} ejemplos a {output_path}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "stats":
        get_stats()

    elif cmd == "export":
        format_type = sys.argv[2] if len(sys.argv) > 2 else "alpaca"
        if format_type == "alpaca":
            export_alpaca()
        else:
            print(f"Formato no soportado: {format_type}")

    elif cmd == "add" and len(sys.argv) >= 4:
        context = sys.argv[2]
        response = sys.argv[3]
        add_example("general", "Responde de manera útil y directa.", context, response)

    elif cmd == "add-curiosity" and len(sys.argv) >= 4:
        add_curiosity_example(sys.argv[2], sys.argv[3])

    elif cmd == "add-reflection" and len(sys.argv) >= 4:
        add_reflection_example(sys.argv[2], sys.argv[3])

    elif cmd == "add-challenge" and len(sys.argv) >= 4:
        add_challenge_example(sys.argv[2], sys.argv[3])

    elif cmd == "add-personality" and len(sys.argv) >= 4:
        add_personality_example(sys.argv[2], sys.argv[3])

    elif cmd == "add-initiative" and len(sys.argv) >= 4:
        add_initiative_example(sys.argv[2], sys.argv[3])

    else:
        print(__doc__)

if __name__ == "__main__":
    main()
