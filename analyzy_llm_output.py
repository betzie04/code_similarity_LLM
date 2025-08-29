import dspy
from openai import OpenAI
import httpx
import csv



def load_assembler_code(input_path, output_path):
    sequences = []
    current = []
    results = []

    with open(input_path, 'r', encoding='utf-8') as f_in, \
     open(output_path, 'w', encoding='utf-8', newline='') as f_out:

        f_out.write("Code Review Results:\n\n")

        lines = f_in.readlines()
        i = 0
        block_count = 0

        while i < len(lines):
            # Suche den Start einer Block-Markierung
            if lines[i].strip() == "------------------------------------------------------------":
                start = i
                i += 1
                # Suche die nächste gestrichelte Linie (Ende des Blocks)
                while i < len(lines) and lines[i].strip() != "------------------------------------------------------------":
                    i += 1
                end = i

                # Inhalt des Blocks extrahieren
                block = lines[start + 1:end]
                block_text = "".join(block)

                # Similarity und Label extrahieren
                similarity = label = None
                for line in block:
                    if line.startswith("Similarity:"):
                        try:
                            similarity = int(line.split("Similarity:")[1].strip())
                        except ValueError:
                            pass
                    elif line.startswith("Label:"):
                        try:
                            label = int(line.split("Label:")[1].strip())
                        except ValueError:
                            pass

                # Bedingung prüfen
                if similarity is not None and label is not None and similarity != label:
                    f_out.write("------------------------------------------------------------\n")
                    f_out.write(block_text)
                    f_out.write("------------------------------------------------------------\n\n")
                    block_count += 1

            i += 1

        print(f"✅ {block_count} relevante Blöcke wurden in '{output_path}' geschrieben.")



    # Jetzt ist sequences eine Liste von Listen, jede innere Liste ist eine Sequenz

if __name__ == "__main__":
    input_path = 'analyse_binary_llm/data-src/similarity_test_llm/llm_results_sim_range.txt'
    output_path = 'analyse_binary_llm/data-src/similarity_test_llm/llm_diff_label_sim.txt'
    load_assembler_code(input_path, output_path)
    