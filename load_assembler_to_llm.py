import dspy
from openai import OpenAI
import httpx


class CodeReviewSignature(dspy.Signature):
    """
    Your task is to analyze te given assembler code, try to figure out if the first code and the second code are similar. 
    Say -1 if they are not similar, and 1 if they are similar. Say also a range of similarity between -1 and 1, where -1 means not similar at all, and 1 means very similar.
    Return only the similarity score as an integer, e.g. -1 or 1, the range of similarity and write a short text, why you consider it to be similar or dissimilar.
    """

    code1:str = dspy.InputField(desc="The first code")
    code2:str = dspy.InputField(desc="The second code")
    #cosine_similarity:float = dspy.OutputField(desc="cosine similarity between the first code and the second code")
    similarity_score:float = dspy.OutputField(desc="similarity score between the first code and the second code, in range [-1, 1]")
    similarity:int = dspy.OutputField(desc="similarity score of the first code and the second code")
    explanation:str = dspy.OutputField(desc="explanation of the similarity score")

    #severity:list[int] = dspy.OutputField(desc="The severity score of the vuln")





def load_assembler_code(path):
    sequences = []
    current = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current:
                    sequences.append(current)
                    current = []
            else:
                current.append(line)
        if current:  # letzte Sequenz anhängen, falls Datei nicht mit Leerzeile endet
            sequences.append(current)

        return sequences
    # Jetzt ist sequences eine Liste von Listen, jede innere Liste ist eine Sequenz

if __name__ == "__main__":
    sequences_1 = load_assembler_code('data-src/similarity_test_llm/llm_input.input0')
    sequences_2 = load_assembler_code('data-src/similarity_test_llm/llm_input.input1') 
    labels = load_assembler_code('data-src/similarity_test_llm/llm_input.label')
    min_length = min(len(sequences_1[0]), len(sequences_2[0]))
    print(len(sequences_1[0]), len(sequences_2[0]), len(labels[0]))
    print(len(sequences_1), len(sequences_2), len(labels))

        # Filtere leere Sequenzen heraus
    print(f"Loaded {len(sequences_2)} sequences of assembler code.")
    model = "DeepSeek-R1-05"
    #model = "/root/models/DeepSeek-R1-Distill-Llama-70B"

    base_url = "https://10.244.192.5:32001/v1/"
    api_key = "None"

    client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=httpx.Client(verify=False)
            )

    lm = dspy.LM(

                model=f"openai/{model}",
                api_key=api_key,
                max_tokens=2000,
                temperature=0.6,
                top_p=0.95,
                cache=False,
                client=client
            )
    dspy.configure(lm=lm)
    correct = 0
    incorrect = 0
    #code1= "push rbp mov rbp , rsp sub rsp , 0x30 mov qword ptr [ rbp - 0x18 ] , rdi mov qword ptr [ rbp - 0x20 ] , rsi mov dword ptr [ rbp - 0x24 ] , edx mov rax , qword ptr [ rbp - 0x20 ] mov qword ptr [ rbp - 0x10 ] "
    #code2= "push nop rbp nop mov nop nop nop nop nop rbp nop nop nop , rsp sub nop nop nop nop nop nop nop nop nop rsp nop , 0x30 nop nop nop nop mov qword nop nop nop nop ptr [ rbp - 0x18 ] nop nop nop , rdi mov qword ptr [ rbp - 0x20 ] , rsi mov dword ptr [ rbp - 0x24 ] , edx mov rax , qword ptr [ rbp - 0x20 ] mov qword ptr [ rbp - 0x10 ] "
    #qa = dspy.Predict(CodeReviewSignature)
    #response = qa(code1=code1, code2=code2)

    #print(f"Similarity: {response.similarity}\nExplanation: {response.explanation}\n")
    #exit()
    output_path = "data-src/similarity_test_llm/llm_results_sim_range.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Code Review Results:\n")
        #f.flush()
        for seq_1, seq_2, label in zip(sequences_1[0][:min_length], sequences_2[0][:min_length], labels[0][:min_length]):
            code1 = seq_1
            code2 = seq_2
            try:
                qa = dspy.Predict(CodeReviewSignature)
                response = qa(code1=code1, code2=code2)
                f.write(f"Code1: {code1}\n")
                f.write(f"Code2: {code2}\n")
                f.write(f"Similarity: {response.similarity}\n")
                f.write(f"Label: {label}\n")
                f.write(f"Similarity Score: {response.similarity_score}\n")
                f.write(f"Explanation: {response.explanation}\n")
                #f.write(f"Cosine Similarity: {response.cosine_similarity}\n")
                f.write("-" * 60 + "\n")  # Trennlinie für Lesbarkeit
                #f.flush()
                is_correct = response.similarity == int(label)
                print(f"Code1: {code1[:20]}\nCode2: {code2[:20]}\nSimilarity: {response.similarity}\nLabel: {label}\n")
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1
            except Exception as e:
                print(f"Error processing code1: {code1[:20]} or code2: {code2[:20]}. Error: {e}")
                f.write(f"Error processing code1: {code1[:20]} or code2: {code2[:20]}. Error: {e}\n")
                f.write("-" * 60 + "\n")  # Trennlinie für Lesbarkeit
                continue
            if (correct + incorrect) % 100 == 0:
                f.write("\nZusammenfassung:\n")
                f.write(f"Richtig: {correct}\n")
                f.write(f"Falsch: {incorrect}\n")
                f.write(f"Genauigkeit: {correct / (correct + incorrect):.2%}\n")
       
        f.write("\nZusammenfassung:\n")
        f.write(f"Richtig: {correct}\n")
        f.write(f"Falsch: {incorrect}\n")
        f.write(f"Genauigkeit: {correct / (correct + incorrect):.2%}\n")
        #f.flush()

    #for seq in sequences:
    #    code1 = "\n".join(seq)
    #    qa = dspy.Predict(CodeReviewSignature)
    #    response = qa(code1=code1)
    #    print(f"Code: {code}\nVulnerabilities: {response.vuls}\nSeverity: {response.severity}\n")
