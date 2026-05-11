import os

def process_file(input_file, output_file):
    # Read lines from the input file and remove any trailing newline
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Sort the lines in lexicographical order
    lines.sort()
    # Write sorted lines with serial numbers, starting from 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for index, entity in enumerate(lines, start=0):
            f.write(f"{entity}\t{index}\n")

if __name__ == '__main__':
    base_dir = r'C:\Code_Compiling\01_HKUST-RA\05_foundation_model_for_KG\REDGNN\data\nell_ind'
    entities_input = os.path.join(base_dir, 'entities.txt')
    entities_output = os.path.join(base_dir, 'entities.txt')
    relations_input = os.path.join(base_dir, 'relations.txt')
    relations_output = os.path.join(base_dir, 'relations.txt')
    
    process_file(entities_input, entities_output)
    process_file(relations_input, relations_output)
