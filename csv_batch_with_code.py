import pandas as pd
import os

input_folder = 'pubchem2'
output_file = 'pubchem_output_2.xlsx'

all_data = pd.DataFrame(columns=['SMILES', 'IC50', 'Source'])

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        print(f"Обробляється файл: {filename}")

        try:
            data = pd.read_csv(file_path)
            data.columns = data.columns.str.strip()
            data = data.dropna(subset=['PUBCHEM_EXT_DATASOURCE_SMILES', 'Standard Value'])
            data = data[data['PUBCHEM_EXT_DATASOURCE_SMILES'] != '']

            if 'PUBCHEM_EXT_DATASOURCE_SMILES' in data.columns and 'Standard Value' in data.columns:
                output_data = data[['PUBCHEM_EXT_DATASOURCE_SMILES', 'Standard Value']]
                output_data.columns = ['SMILES', 'IC50']

                source_name = os.path.splitext(filename)[0]
                output_data['Source'] = source_name

                all_data = pd.concat([all_data, output_data], ignore_index=True)
                print(f"Файл {filename} оброблено успішно.")
            else:
                print(f"У файлі {filename} відсутні необхідні стовпці.")

        except Exception as e:
            print(f"Помилка при обробці файлу {filename}: {e}")

all_data.to_excel(output_file, index=False)
print(f"Всі дані записано в файл {output_file}")
