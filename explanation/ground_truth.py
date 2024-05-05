import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def construct_ground_truth(dc_row, dd, cc, dc):
    # For cells in one of the edge dc_row DC, find cells similar to this cell in CC and return to the list
    sim_cellines1 = cc.loc[(cc['cell1'] == dc_row['cell']), 'cell2'].to_list()
    sim_cellines2 = cc.loc[(cc['cell2'] == dc_row['cell']), 'cell1'].to_list()
    similar_cell_lines = sim_cellines1 + sim_cellines2
    print('len(similar_cell_lines)',len(similar_cell_lines))

    sim_drugs1 = dd.loc[(dd['drug1'] == dc_row['drug']), 'drug2'].to_list()
    sim_drugs2 = dd.loc[(dd['drug2'] == dc_row['drug']), 'drug1'].to_list()
    similar_drugs = sim_drugs1 + sim_drugs2
    print('len(similar_drugs)', len(similar_drugs))

    # Create a set to keep track of added combinations
    added_combinations = set()

    # Create a DataFrame to hold g round truth data
    ground_truth_data = []

    # Check for dc entries that match the original drug and similar cell lines
    for cell_line in similar_cell_lines:
        matched_entries = dc.loc[(dc['drug'] == dc_row['drug']) & (dc['cell'] == cell_line) & (dc['rel'] == dc_row['rel'])]
        for _, matched_row in matched_entries.iterrows():
            ground_truth_data.append((matched_row['cell'],matched_row['rel'],dc_row['drug']))
            added_combinations.add((matched_row['drug'], dc_row['cell']))

    # Check for dc entries that match similar drugs and the original cell line
    for drug in similar_drugs:
        matched_entries = dc.loc[(dc['drug'] == drug) & (dc['cell'] == dc_row['cell']) & (dc['rel'] == dc_row['rel'])]
        for _, matched_row in matched_entries.iterrows():
            ground_truth_data.append((dc_row['cell'],matched_row['rel'],matched_row['drug']))
            added_combinations.add((matched_row['drug'], dc_row['cell']))

    # Check for dc entries that match similar drugs and similar cell lines
    for drug in similar_drugs:
        for cell_line in similar_cell_lines:
            matched_entries = dc.loc[(dc['drug'] == drug) & (dc['cell'] == cell_line) & (dc['rel'] == dc_row['rel'])]
            for _, matched_row in matched_entries.iterrows():
                ground_truth_data.append((matched_row['cell'],matched_row['rel'],matched_row['drug']))
                added_combinations.add((matched_row['drug'], matched_row['cell']))

    # Check for similar drug relationships with original and similar cell lines
    for drug in similar_drugs:
        # Check against original cell line
        if any(dc.loc[(dc['drug'] == drug) & (dc['cell'] == dc_row['cell']), 'rel'] == dc_row['rel']):
            combination = (dc_row['drug'], drug)
            if combination not in added_combinations:
                ground_truth_data.append((dc_row['drug'], 2,drug))  # 2 indicates similarity
                added_combinations.add(combination)
        # Check against similar cell lines
        for cell_line in similar_cell_lines:
            if any(dc.loc[(dc['drug'] == drug) & (dc['cell'] == cell_line), 'rel'] == dc_row['rel']):
                combination = (dc_row['drug'], drug)
                if combination not in added_combinations:
                    ground_truth_data.append((dc_row['drug'], 2, drug))  # 2 indicates similarity
                    added_combinations.add(combination)

    # Check for similar cell line relationships with original and similar drugs
    for cell_line in similar_cell_lines:
        # Check against original drug
        if any(dc.loc[(dc['drug'] == dc_row['drug']) & (dc['cell'] == cell_line), 'rel'] == dc_row['rel']):
            combination = (dc_row['cell'], cell_line)
            if combination not in added_combinations:
                ground_truth_data.append((dc_row['cell'], 3 ,cell_line))  # 3 indicates similarity
                added_combinations.add(combination)
        # Check against similar drugs
        for drug in similar_drugs:
            if any(dc.loc[(dc['drug'] == drug) & (dc['cell'] == cell_line), 'rel'] == dc_row['rel']):
                combination = (dc_row['cell'], cell_line)
                if combination not in added_combinations:
                    ground_truth_data.append((dc_row['cell'], 3, cell_line))  # 3 indicates similarity
                    added_combinations.add(combination)

    return ground_truth_data

def load_and_preprocess_data():
    dc = pd.read_csv('../prediction/data/resopnse_triples.csv')
    dc = dc.sample(frac=0.008, random_state=42)
    cc = pd.read_csv('../prediction/data/cell_similar_triples.csv')
    dd = pd.read_csv('../prediction/data/drug_similar_triples.csv')

    dc.columns = ['cell', 'rel', 'drug']
    cc.columns = ['cell1', 'rel', 'cell2']
    dd.columns = ['drug1', 'rel', 'drug2']

    return dc, cc, dd
def process_chunk(chunk, dd, cc, dc):
    ground_truth_rows = []
    for _, row in tqdm(chunk.iterrows(), total=chunk.shape[0]):
        ground_truth = construct_ground_truth(row, dd, cc, dc)
        flat_ground_truth = [item for sublist in ground_truth for item in sublist]
        ground_truth_rows.append([row['cell'], row['rel'], row['drug']] + flat_ground_truth)
    return ground_truth_rows
def main():
    dc, cc, dd = load_and_preprocess_data()

    num_chunks = 5
    chunks = np.array_split(dc, num_chunks)

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_chunk, chunks, [dd]*num_chunks, [cc]*num_chunks, [dc]*num_chunks), total=num_chunks))

    ground_truth_rows = []
    for result in results:
        ground_truth_rows.extend(result)

    max_columns = max(len(row) for row in ground_truth_rows)
    column_names = ['obj', 'rel', 'sbj'] + [f'GT_{i}' for i in range(1, max_columns-2)]

    ground_truth_df = pd.DataFrame(ground_truth_rows, columns=column_names)
    ground_truth_df.to_csv('data/gt_all.csv', index=False)

if __name__ == "__main__":
    main()


